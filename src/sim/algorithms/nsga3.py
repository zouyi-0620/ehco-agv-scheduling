"""NSGA-III baseline (SPEC.md section 8): 4 objectives, Das-Dennis H=5 (56 ref
points), equal 20,000-eval budget.

Environment selection follows Deb & Jain (2014): non-dominated sorting, then the
last partially-filled front is completed by reference-line niche counts. As a
documented simplification (SPEC D12) the per-generation normalisation is
ideal-shift + per-dimension range scaling instead of intercept-based scaling.
"""
from __future__ import annotations

import time

import numpy as np

from .. import constants as C
from ..metrics import das_dennis_ref_points, fast_non_dominated_sort
from ..objectives import EvalConfig, evaluate_population, knee_point_selection
from ..scenario import Scenario
from .common import (AlgorithmResult, EvalBudget, eligible_mask,
                     finite_F, init_population, mutate, pc_at, pm_at,
                     repair, tournament, uniform_col_crossover_pair)

NAME = "NSGA-III"
OBJ_IDX = (0, 1, 2, 3)


def _ideal_scale_normalize(Fsub: np.ndarray) -> np.ndarray:
    """Shift by ideal point, scale by per-dim range of the shifted set."""
    zmin = Fsub.min(axis=0)
    Fsh = Fsub - zmin
    span = Fsh.max(axis=0)
    span = np.where(span < 1e-12, 1.0, span)
    return Fsh / span


def _associate(Fn: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest reference line (unit direction) per point.

    Returns (ref_id, distance) arrays of length n.
    """
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)      # (r, m)
    proj = Fn @ Wn.T                                       # (n, r)
    resid = Fn[:, None, :] - proj[:, :, None] * Wn[None, :, :]
    dist = np.linalg.norm(resid, axis=2)                   # (n, r)
    j = np.argmin(dist, axis=1)
    return j, dist[np.arange(len(dist)), j]


def _niche_fill(members: np.ndarray, assoc: np.ndarray, dist: np.ndarray,
                rho: np.ndarray, need: int) -> list[int]:
    """Complete the last front: repeatedly pick the member of the least-filled
    reference line closest to it (NSGA-III niche preservation)."""
    m = list(range(len(members)))
    selected: list[int] = []
    rho = rho.copy()
    while len(selected) < need and m:
        cand_refs = assoc[m]
        j = int(cand_refs[np.argmin(rho[cand_refs])])
        cand_j = [i for i in m if assoc[i] == j]
        i_best = min(cand_j, key=lambda i: dist[i])
        selected.append(int(members[i_best]))
        rho[j] += 1
        m.remove(i_best)
    return selected


def run(sc: Scenario, seed: int, cfg: EvalConfig | None = None,
        budget: EvalBudget | None = None) -> AlgorithmResult:
    cfg = cfg or EvalConfig()
    budget = budget or EvalBudget()
    rng = np.random.default_rng(seed)
    N = len(sc.agvs)
    M = len(sc.tasks)
    P = C.NP
    oi = list(OBJ_IDX)
    h_cum = np.array([a.h0 for a in sc.agvs])
    elig = eligible_mask(h_cum, cfg)

    W = das_dennis_ref_points(C.NSGA3_H, len(OBJ_IDX))     # (56, 4)
    n_ref = len(W)

    pop = init_population(P, M, elig, rng)
    budget.spend(P)
    ev = evaluate_population(pop, sc, cfg, h_cum)
    F = np.stack([ev["f1"], ev["f2"], ev["f3"], ev["f4"]], axis=1)
    Fs = finite_F(F)

    front_history: list[tuple[int, np.ndarray]] = []
    start = time.perf_counter()
    g = 0
    while budget.can_afford(P):
        # mating: tournament on front rank (tie -> random via second=None)
        ranks = np.zeros(P, dtype=int)
        frs = fast_non_dominated_sort(Fs[:, oi])
        for k, fr in enumerate(frs):
            ranks[fr] = k
        off = np.empty((P, M), dtype=int)
        pc, pm = pc_at(g), pm_at(g)
        for i in range(0, P, 2):
            a = tournament(2, ranks, None, rng, 2)
            p1, p2 = pop[a[0]], pop[a[1]]
            if rng.random() < pc:
                c1, c2 = uniform_col_crossover_pair(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            mutate(c1, elig, rng, pm)
            mutate(c2, elig, rng, pm)
            off[i] = c1
            if i + 1 < P:
                off[i + 1] = c2
        repair(off, elig, rng)

        budget.spend(P)
        ev2 = evaluate_population(off, sc, cfg, h_cum)
        F2 = np.stack([ev2["f1"], ev2["f2"], ev2["f3"], ev2["f4"]], axis=1)
        F2s = finite_F(F2)

        comb_pop = np.vstack([pop, off])
        comb_F = np.vstack([F, F2])
        comb_s = finite_F(comb_F)

        # ---- environmental selection -------------------------------------
        Fn = _ideal_scale_normalize(comb_s[:, oi])
        assoc, dist = _associate(Fn, W)
        rho = np.zeros(n_ref, dtype=int)
        fronts = fast_non_dominated_sort(comb_s[:, oi])
        sel: list[int] = []
        for fr in fronts:
            if len(sel) + len(fr) <= P:
                sel.extend(int(x) for x in fr)
                rho += np.bincount(assoc[fr], minlength=n_ref)
            else:
                need = P - len(sel)
                if need > 0:
                    sel.extend(_niche_fill(fr, assoc[fr], dist[fr], rho, need))
                break
        sel_arr = np.asarray(sel, dtype=int)
        pop = comb_pop[sel_arr]
        F = comb_F[sel_arr]
        Fs = comb_s[sel_arr]

        if g % 10 == 0:
            front_history.append((g, F.copy()))
        g += 1

    wall = time.perf_counter() - start
    fronts_final = fast_non_dominated_sort(Fs[:, oi])
    f1_idx = fronts_final[0]
    sub = Fs[f1_idx][:, oi]
    fin_rows = np.isfinite(F[f1_idx][:, oi]).all(axis=1)
    if fin_rows.any():
        k = int(np.where(fin_rows)[0][int(knee_point_selection(sub[fin_rows]))])
    else:
        k = int(np.argmin(sub.sum(axis=1)))
    kidx = int(f1_idx[k])

    params = {"np": P, "gmax": C.GMAX, "budget": budget.limit,
              "obj_idx": OBJ_IDX, "H": C.NSGA3_H, "n_ref": n_ref,
              "normalization": "ideal+range", "gens_actual": g,
              "evals": budget.spent}
    return AlgorithmResult(
        name=NAME, seed=seed, assign_final=pop, F_final=F,
        obj_idx=OBJ_IDX, fronts=fronts_final, knee_idx=kidx,
        best_assign=pop[kidx].copy(), best_F=F[kidx],
        evals=budget.spent, wall_s=wall,
        front_history=front_history, params=params)
