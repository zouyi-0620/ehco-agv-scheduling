"""Algorithm-level metrics and statistics (SPEC.md sections 9, 11).

Contents
--------
- fast_non_dominated_sort      : Deb's O(N^2) non-dominated sort (minimisation)
- crowding_distance            : NSGA-II crowding distance on an objective subset
- das_dennis_ref_points        : Das-Dennis simplex-lattice reference points
- hypervolume                  : exact HV via recursive slicing (any dimension,
                                 efficient for m <= 4, fronts <= a few hundred)
- gini                         : Gini coefficient of a degradation vector
- paired_wilcoxon / holm_adjust / cohen_dz / ci95
                               : D9 statistical pipeline (Wilcoxon + Holm,
                                 Cohen's d_z, Student t 95% CI)

All objective values are minimised. HV reference point and normalisation follow
decision point D5 (min-max over the union of all algorithms' final fronts within
a run, reference point (1, 1, ..., 1) after normalisation); the experiments
layer computes the union normalisation - this module only provides the raw HV
of a (already normalised / clipped) front.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _sps


# --------------------------------------------------------------------------
# Pareto / front utilities (minimisation)
# --------------------------------------------------------------------------

def dominates(p: np.ndarray, q: np.ndarray) -> bool:
    """True if p strictly dominates q (p <= q in all dims, < in at least one)."""
    return bool(np.all(p <= q) and np.any(p < q))


def fast_non_dominated_sort(F: np.ndarray) -> list[np.ndarray]:
    """Deb et al. (2002) non-dominated sort, vectorised pairwise dominance.

    F : (N, m) objective matrix (finite; callers sanitise inf).
    Returns list of index arrays front_0 (Pareto), front_1, ...

    Complexity: O(N^2 m) vectorised comparisons + front peeling (each front
    removes the points with zero remaining dominators).
    """
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    if n == 0:
        return []
    le = (F[:, None, :] <= F[None, :, :]).all(axis=2)   # i <= j elementwise
    lt = (F[:, None, :] < F[None, :, :]).any(axis=2)    # i < j in some dim
    dom = le & lt                                        # i dominated by j
    if n:
        np.fill_diagonal(dom, False)
    remaining = np.ones(n, dtype=bool)
    fronts: list[np.ndarray] = []
    while remaining.any():
        nd = dom[remaining, :].sum(axis=0)               # dominators still left
        cur = remaining & (nd == 0)
        fronts.append(np.flatnonzero(cur))
        remaining[cur] = False
    return fronts


def front_ranks(F: np.ndarray) -> np.ndarray:
    """Per-row front index (0 = Pareto front) of a finite objective matrix."""
    fronts = fast_non_dominated_sort(F)
    r = np.full(F.shape[0], len(fronts), dtype=int)
    for k, fr in enumerate(fronts):
        r[fr] = k
    return r


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance per point (higher = more isolated)."""
    F = np.asarray(F, dtype=float)
    n, m = F.shape
    if n == 0:
        return np.zeros(0)
    if n <= 2:
        return np.full(n, np.inf)
    cd = np.zeros(n)
    for k in range(m):
        order = np.argsort(F[:, k])
        fmin, fmax = F[order[0], k], F[order[-1], k]
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf
        if fmax - fmin < 1e-12:
            continue
        span = fmax - fmin
        # interior points: gap between neighbours / span
        cd[order[1:-1]] += (F[order[2:], k] - F[order[:-2], k]) / span
    return cd


# --------------------------------------------------------------------------
# Reference points / weight vectors
# --------------------------------------------------------------------------

def das_dennis_ref_points(H: int, m: int) -> np.ndarray:
    """Das-Dennis simplex lattice: all compositions of H into m parts / H.

    Returns (C(H+m-1, m-1), m) rows summing to 1.
    """
    pts: list[list[int]] = []

    def rec(rem: int, dims_left: int, cur: list[int]) -> None:
        if dims_left == 1:
            pts.append(cur + [rem])
            return
        for i in range(rem + 1):
            rec(rem - i, dims_left - 1, cur + [i])

    rec(H, m, [])
    W = np.asarray(pts, dtype=float) / float(H)
    return W


def dirichlet_weights(n: int, m: int, seed: int) -> np.ndarray:
    """`n` uniform-over-simplex weight vectors (Dirichlet(1,...,1)), seeded.

    Used for MOEA/D when no integer Das-Dennis division yields exactly `n`
    vectors (SPEC decision point D12: 100 vectors, no H gives 100 for m=4).
    """
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(m), size=n)


# --------------------------------------------------------------------------
# Exact hypervolume (recursive slicing)
# --------------------------------------------------------------------------

def _nd_filter(F: np.ndarray) -> np.ndarray:
    """Indices of points not dominated by any other point (minimisation)."""
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    keep = []
    for i in range(n):
        dom = False
        for j in range(n):
            if i != j and dominates(F[j], F[i]):
                dom = True
                break
        if not dom:
            keep.append(i)
    return np.asarray(keep, dtype=int)


def _hv_rec(F: np.ndarray, ref: np.ndarray) -> float:
    """Recursive-slicing HV of a clipped, non-dominated point set."""
    m = len(ref)
    n = F.shape[0]
    if n == 0:
        return 0.0
    if m == 1:
        return float(max(0.0, ref[0] - float(np.min(F[:, 0]))))
    last = F[:, m - 1]
    vals = np.unique(last)
    vol = 0.0
    for k in range(len(vals)):
        v = vals[k]
        h = (vals[k + 1] - v) if k + 1 < len(vals) else (ref[m - 1] - v)
        if h <= 0.0:
            continue
        sub_all = F[last <= v, : m - 1]
        if len(sub_all) == 0:
            continue
        sub = sub_all[_nd_filter(sub_all)]
        if len(sub) == 0:
            continue
        vol += h * _hv_rec(sub, ref[: m - 1])
    return vol


def hypervolume(F: np.ndarray, ref: np.ndarray,
                obj_idx: tuple[int, ...] | None = None) -> float:
    """Exact hypervolume of the union of boxes [p, ref] over points p in F.

    F   : (N, m_full) or (N, m) objective matrix (minimisation)
    ref : reference point (per-dimension, same length as selected dims)
    obj_idx : optional subset of columns of F to use.

    Points with any coordinate above ref are clipped (their box is empty along
    that dimension). Dominated points are filtered first.
    """
    F = np.asarray(F, dtype=float)
    if obj_idx is not None:
        F = F[:, list(obj_idx)]
        ref = np.asarray(ref, dtype=float)[list(obj_idx)]
    ref = np.asarray(ref, dtype=float)
    if F.ndim == 1:
        F = F.reshape(1, -1)
    if F.shape[0] == 0:
        return 0.0
    F = np.minimum(F, ref)                 # clip above reference
    idx = _nd_filter(F)
    F = F[idx]
    if len(F) == 0:
        return 0.0
    return _hv_rec(F, ref)


def gini(x: np.ndarray) -> float:
    """Gini coefficient of a (non-negative) vector."""
    x = np.sort(np.abs(np.asarray(x, dtype=float)))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum() / (n * x.sum())) - (n + 1) / n)


# --------------------------------------------------------------------------
# D9 statistical pipeline
# --------------------------------------------------------------------------

def paired_wilcoxon(a: np.ndarray, b: np.ndarray,
                    alternative: str = "less") -> float:
    """Paired one-sided Wilcoxon signed-rank p-value (a better than b?).

    `alternative="less"` tests median(a) < median(b) (lower objective = better).
    Pratt zero handling avoids discarding tied pairs.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    res = _sps.wilcoxon(a, b, zero_method="pratt", alternative=alternative)
    return float(res.pvalue)


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (monotone non-decreasing)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, oi in enumerate(order):
        v = min(1.0, (m - i) * p[oi])
        running = max(running, v)
        adj[oi] = running
    return adj


def cohen_dz(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d_z = mean(diff) / sd(diff)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    s = d.std(ddof=1)
    if s < 1e-12:
        return float("nan") if abs(d.mean()) > 1e-12 else 0.0
    return float(d.mean() / s)


def ci95(x: np.ndarray) -> tuple[float, float]:
    """Student-t 95% CI of the mean (df = n-1)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else float("nan"),
                float(x.mean()) if n else float("nan"))
    se = x.std(ddof=1) / np.sqrt(n)
    t = _sps.t.ppf(0.975, df=n - 1)
    return (float(x.mean() - t * se), float(x.mean() + t * se))
