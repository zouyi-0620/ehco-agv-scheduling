"""Shared algorithm infrastructure (SPEC.md sections 7, 8).

Common pieces for the eight comparison algorithms:
- equal evaluation budget (20,000 evals) enforced by :class:`EvalBudget`
- D3 genetic operators: uniform column-wise crossover, swap + per-gene mutation
- linear pc/pm schedule (0.9->0.6 / 0.05->0.2 over Gmax=200)
- an NSGA-II-style generational core whose *last-front fill criterion* is
  pluggable:
    * crowding distance            (NSGA-II 2/3-obj)
    * FLC / LWC weighted scalar    (AW-NSGA-II / LWC)  [变权标量化, SPEC 7]
    * single-objective value       (A*-GA)
- a uniform :class:`AlgorithmResult` dataclass consumed by experiments.

Feasibility: an individual is repaired so no task is assigned to an ineligible
AGV (ineligible = h_cum below the hard threshold of the EvalConfig variant).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .. import constants as C
from ..flc import flc_weights, scenario_features
from ..metrics import (crowding_distance, fast_non_dominated_sort,
                       front_ranks)
from ..objectives import EvalConfig, evaluate_population, knee_point_selection
from ..scenario import Scenario

BUDGET_DEFAULT = C.NP * C.GMAX            # 20,000 evals (SPEC section 8)
LARGE = 1e9                               # sanitised value for inf objectives


@dataclass
class EvalBudget:
    """Equal-budget bookkeeping: at most `limit` evaluations per run."""
    limit: int = BUDGET_DEFAULT
    spent: int = 0

    def can_afford(self, n: int) -> bool:
        return self.spent + n <= self.limit

    def spend(self, n: int) -> None:
        if not self.can_afford(n):
            raise ValueError(
                f"budget exhausted: spent {self.spent}, asked {n}, limit {self.limit}")
        self.spent += n


@dataclass
class AlgorithmResult:
    """Uniform output of every algorithm run."""
    name: str
    seed: int
    assign_final: np.ndarray          # (P, M) final population
    F_final: np.ndarray               # (P, 4) full objectives of final population
    obj_idx: tuple[int, ...]          # objective subspace the algorithm optimised
    fronts: list[np.ndarray]          # ND fronts of final pop (obj subspace)
    knee_idx: int                     # row index of the reported solution
    best_assign: np.ndarray           # (M,) reported solution assignment
    best_F: np.ndarray                # (4,) full objectives of reported solution
    evals: int
    wall_s: float
    front_history: list[tuple[int, np.ndarray]] = field(default_factory=list)
    params: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# operators (D3)
# --------------------------------------------------------------------------

def pc_at(g: int) -> float:
    return C.PC_MAX - (C.PC_MAX - C.PC_MIN) * g / C.GMAX      # 0.9 -> 0.6


def pm_at(g: int) -> float:
    return C.PM_MIN + (C.PM_MAX - C.PM_MIN) * g / C.GMAX      # 0.05 -> 0.2


def eligible_mask(h_cum: np.ndarray, cfg: EvalConfig) -> np.ndarray:
    """AGVs that may receive tasks under the variant's hard constraint."""
    if cfg.hard_exclusion_only:
        return h_cum >= C.H_SAFE
    if not cfg.use_health_objective:
        return np.ones(len(h_cum), dtype=bool)
    return h_cum >= C.H_CRIT


def init_population(P: int, M: int, eligible: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
    e_idx = np.where(eligible)[0]
    if len(e_idx) == 0:
        raise ValueError("no eligible AGV for initialisation")
    a = rng.integers(0, len(e_idx), size=(P, M))
    return e_idx[a]


def repair(assign: np.ndarray, eligible: np.ndarray,
           rng: np.random.Generator) -> np.ndarray:
    """Re-assign every gene on an ineligible AGV to a random eligible one."""
    e_idx = np.where(eligible)[0]
    bad = ~eligible[assign]
    if bad.any():
        assign[bad] = e_idx[rng.integers(0, len(e_idx), size=int(bad.sum()))]
    return assign


def uniform_col_crossover_pair(p1: np.ndarray, p2: np.ndarray,
                               rng: np.random.Generator
                               ) -> tuple[np.ndarray, np.ndarray]:
    """D3: uniform crossover per task column (PMX spirit on the new encoding)."""
    mask = rng.random(p1.shape[0]) < 0.5
    c1 = np.where(mask, p1, p2)
    c2 = np.where(mask, p2, p1)
    return c1, c2


def mutate(assign: np.ndarray, eligible: np.ndarray,
           rng: np.random.Generator, pm: float) -> np.ndarray:
    """D3: swap two tasks' AGV + per-gene reassignment with prob pm."""
    M = len(assign)
    if M >= 2:
        i, j = rng.choice(M, size=2, replace=False)
        tmp = assign[i]
        assign[i] = assign[j]
        assign[j] = tmp
    hit = rng.random(M) < pm
    if hit.any():
        e_idx = np.where(eligible)[0]
        assign[hit] = e_idx[rng.integers(0, len(e_idx), size=int(hit.sum()))]
    return assign


def tournament(k: int, ranks: np.ndarray, second: np.ndarray | None,
               rng: np.random.Generator, size: int) -> np.ndarray:
    """k-ary tournament; lower rank (then lower `second`) wins.

    Pass second = -crowding to keep the most isolated points (NSGA-II),
    or second = scalar/objective value to minimise it directly.
    """
    idx = rng.integers(0, len(ranks), size=(size, k))
    out = np.empty(size, dtype=int)
    for t in range(size):
        cand = idx[t]
        r = ranks[cand]
        s = np.zeros(k) if second is None else second[cand]
        out[t] = cand[np.lexsort((s, r))[0]]
    return out


def finite_F(F: np.ndarray) -> np.ndarray:
    """Replace inf (infeasible) objectives with a large finite value."""
    return np.where(np.isinf(F), LARGE, F)


def normalize_minmax(Fsub: np.ndarray) -> np.ndarray:
    """Per-column min-max to [0, 1]; constant columns -> 0.5 (SPEC 7)."""
    Fsub = np.asarray(Fsub, dtype=float)
    out = np.empty_like(Fsub)
    for k in range(Fsub.shape[1]):
        lo, hi = Fsub[:, k].min(), Fsub[:, k].max()
        if hi - lo > 1e-12:
            out[:, k] = (Fsub[:, k] - lo) / (hi - lo)
        else:
            out[:, k] = 0.5
    return out


# --------------------------------------------------------------------------
# evaluation wrapper
# --------------------------------------------------------------------------

def _eval(assign: np.ndarray, sc: Scenario, cfg: EvalConfig,
          h_cum: np.ndarray, budget: EvalBudget) -> dict:
    n = assign.shape[0]
    budget.spend(n)
    return evaluate_population(assign, sc, cfg, h_cum)


def _stack_F(ev: dict) -> np.ndarray:
    return np.stack([ev["f1"], ev["f2"], ev["f3"], ev["f4"]], axis=1)


# --------------------------------------------------------------------------
# NSGA-II-style generational core (shared by AW-NSGA-II, NSGA-II, LWC, A*-GA)
# --------------------------------------------------------------------------

def nsga2_core(sc: Scenario, seed: int, obj_idx: tuple[int, ...],
               cfg: EvalConfig | None = None,
               budget: EvalBudget | None = None,
               name: str = "NSGA-II",
               weight_fn=None,          # h_cum -> (4,) weights (变权标量化)
               use_crowding: bool = False,
               params_extra: dict | None = None,
               pop_size: int | None = None) -> AlgorithmResult:
    """Generational NSGA-II with pluggable last-front fill criterion.

    weight_fn is used for the SPEC-7 "变权标量化" last-front fill (keep lowest
    s(x) = sum w_k f̃_k); when None and use_crowding is False, the last front
    is filled by the first objective of `obj_idx` (single-objective mode).

    Equal budget: init (P evals) + as many P-eval generations as fit.
    pop_size overrides the population size (E4 local replanning P=50 with
    a 4,000-eval budget => Gref=80; the pc/pm ramp then uses Gref instead of
    C.GMAX so the "0.9->0.6 / 0.01->0.02 over the run" schedule stays correct).
    The default path (pop_size=None, budget=20,000) is bit-identical to the
    pre-E4 behaviour (verified against e1_metrics.csv).
    """
    cfg = cfg or EvalConfig()
    budget = budget or EvalBudget()
    rng = np.random.default_rng(seed)
    N = len(sc.agvs)
    M = len(sc.tasks)
    P = C.NP if pop_size is None else pop_size
    local_sched = (pop_size is not None) or (budget.limit != C.NP * C.GMAX)
    Gref = (budget.limit // P) if local_sched else C.GMAX
    h_cum = np.array([a.h0 for a in sc.agvs])
    elig = eligible_mask(h_cum, cfg)
    oi = list(obj_idx)

    pop = init_population(P, M, elig, rng)
    ev = _eval(pop, sc, cfg, h_cum, budget)
    F = _stack_F(ev)
    Fs = finite_F(F)

    front_history: list[tuple[int, np.ndarray]] = []
    params = {"np": P, "gmax": Gref, "budget": budget.limit,
              "obj_idx": obj_idx, "use_crowding": use_crowding,
              "pc_schedule": (C.PC_MAX, C.PC_MIN), "pm_schedule": (C.PM_MIN, C.PM_MAX)}
    if params_extra:
        params.update(params_extra)
    start = time.perf_counter()
    g = 0
    while budget.can_afford(P):
        # ---- current-generation selection criteria ----------------------
        if weight_fn is not None:
            w = np.asarray(weight_fn(sc, h_cum), dtype=float)
        else:
            w = None
        if use_crowding and w is None:
            # pure crowding tournament (no weight guidance)
            second = -crowding_distance(Fs[:, oi])
        elif w is not None:
            # weight-guided tournament (FLC scalarisation); crowding is used
            # for the environmental-selection last-front fill when
            # use_crowding=True (AW-NSGA-II: FLC steers search, crowding
            # preserves front diversity)
            second = normalize_minmax(Fs[:, oi]) @ w[oi]
        else:
            second = Fs[:, oi[0]]
        ranks = front_ranks(Fs[:, oi])

        # ---- offspring ----------------------------------------------------
        off = np.empty((P, M), dtype=int)
        if local_sched:
            pc = C.PC_MAX - (C.PC_MAX - C.PC_MIN) * g / Gref
            pm = C.PM_MIN + (C.PM_MAX - C.PM_MIN) * g / Gref
        else:
            pc, pm = pc_at(g), pm_at(g)
        for i in range(0, P, 2):
            a = tournament(2, ranks, second, rng, 2)
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

        ev2 = _eval(off, sc, cfg, h_cum, budget)
        F2 = _stack_F(ev2)
        F2s = finite_F(F2)
        comb_pop = np.vstack([pop, off])
        comb_F = np.vstack([F, F2])
        comb_s = finite_F(comb_F)

        # ---- environmental selection -------------------------------------
        fronts = fast_non_dominated_sort(comb_s[:, oi])
        sel: list[int] = []
        for fr in fronts:
            if len(sel) + len(fr) <= P:
                sel.extend(int(x) for x in fr)
            else:
                need = P - len(sel)
                if need > 0:
                    if use_crowding:
                        cd = crowding_distance(comb_s[fr][:, oi])
                        order = np.argsort(-cd)
                    elif w is not None:
                        sc_ = normalize_minmax(comb_s[fr][:, oi]) @ w[oi]
                        order = np.argsort(sc_)
                    else:
                        order = np.argsort(comb_s[fr][:, oi[0]])
                    sel.extend(int(x) for x in fr[order[:need]])
                break
        sel_arr = np.asarray(sel, dtype=int)
        pop = comb_pop[sel_arr]
        F = comb_F[sel_arr]
        Fs = comb_s[sel_arr]

        if g % 10 == 0:
            front_history.append((g, F.copy()))
        g += 1

    wall = time.perf_counter() - start
    params["gens_actual"] = g
    params["evals"] = budget.spent

    fronts_final = fast_non_dominated_sort(Fs[:, oi])
    f1_idx = fronts_final[0]
    sub = Fs[f1_idx][:, oi]
    fin_rows = np.isfinite(F[f1_idx][:, oi]).all(axis=1)
    if len(obj_idx) == 1:
        k = int(np.argmin(sub[:, 0]))
    else:
        if fin_rows.any():
            k = int(knee_point_selection(sub[fin_rows]))
            # map back to full-front index
            full_k = int(np.where(fin_rows)[0][k])
            k = full_k
        else:
            k = int(np.argmin(sub.sum(axis=1)))
    kidx = int(f1_idx[k])
    best_F = F[kidx]

    return AlgorithmResult(
        name=name, seed=seed,
        assign_final=pop, F_final=F, obj_idx=obj_idx,
        fronts=fronts_final, knee_idx=kidx,
        best_assign=pop[kidx].copy(), best_F=best_F,
        evals=budget.spent, wall_s=wall,
        front_history=front_history, params=params)


# --------------------------------------------------------------------------
# weight providers
# --------------------------------------------------------------------------

def flc_weight_provider(sc: Scenario, h_cum: np.ndarray) -> np.ndarray:
    """FLC-based adaptive weights (AW-NSGA-II, SPEC section 7)."""
    rho, gamma, h_bar = scenario_features(None, sc.tasks, h_cum)
    return flc_weights(rho, gamma, h_bar)


def lwc_weight_provider(sc: Scenario, h_cum: np.ndarray) -> np.ndarray:
    """Linear weight controller: w4 = a + d*(1-h_bar), rest balanced (SPEC 8)."""
    h_bar = float(np.mean(h_cum))
    w4 = C.LWC_A + C.LWC_D * (1.0 - h_bar)
    w4 = float(np.clip(w4, 0.0, 1.0))
    rest = (1.0 - w4) / 3.0
    return np.array([rest, rest, rest, w4])
