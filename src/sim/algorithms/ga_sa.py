"""GA-SA baseline (SPEC.md section 8): weighted single objective + simulated
annealing refinement; battery (SoH) < 0.6 hard exclusion.

Structure (SPEC D12, documented defaults):
- Phase 1 GA (75% of budget = 15,000 evals): equal-weight scalar
  s(x) = 0.25 * sum f̃_k over min-max normalised objectives, elitist
  (100-population GA with the shared D3 operators).
- Phase 2 SA (remaining 5,000 evals): Metropolis refinement of the GA best,
  T0 = 0.1, cooling factor alpha = 0.999, one swap-neighbour per iteration
  (1 eval each); normalisation fixed from the GA final population for a stable
  scalar scale.
- EvalConfig: hard_exclusion_only -> AGVs with h < 0.6 are ineligible
  (manuscript: "battery SoH < 0.6 hard exclusion"; our combined health h_cum
  proxies battery SoH - documented in SPEC).
"""
from __future__ import annotations

import time

import numpy as np

from .. import constants as C
from ..objectives import EvalConfig, evaluate_population
from ..scenario import Scenario
from .common import (AlgorithmResult, EvalBudget, eligible_mask, finite_F,
                     init_population, mutate, normalize_minmax, repair)

NAME = "GA-SA"
OBJ_IDX = (0, 1, 2, 3)
W_EQ = np.array([0.25, 0.25, 0.25, 0.25])
GA_SHARE = 0.75                     # fraction of budget for the GA phase
SA_T0 = 0.1
SA_ALPHA = 0.999


def _scalar(Fn: np.ndarray) -> np.ndarray:
    return Fn @ W_EQ


def run(sc: Scenario, seed: int, cfg: EvalConfig | None = None,
        budget: EvalBudget | None = None) -> AlgorithmResult:
    if cfg is None:
        cfg = EvalConfig(hard_exclusion_only=True)
    budget = budget or EvalBudget()
    rng = np.random.default_rng(seed)
    N = len(sc.agvs)
    M = len(sc.tasks)
    P = C.NP
    oi = list(OBJ_IDX)
    h_cum = np.array([a.h0 for a in sc.agvs])
    elig = eligible_mask(h_cum, cfg)

    ga_limit = int(budget.limit * GA_SHARE)         # 15,000 evals
    pop = init_population(P, M, elig, rng)
    budget.spend(P)
    ev = evaluate_population(pop, sc, cfg, h_cum)
    F = np.stack([ev["f1"], ev["f2"], ev["f3"], ev["f4"]], axis=1)
    Fs = finite_F(F)

    front_history: list[tuple[int, np.ndarray]] = []
    start = time.perf_counter()
    g = 0
    best_row = int(np.argmin(_scalar(normalize_minmax(Fs[:, oi]))))
    best_so_far = F[best_row].copy()

    # ---------------- Phase 1: elitist GA on the weighted scalar ----------
    while budget.spent + P <= ga_limit:
        s = _scalar(normalize_minmax(Fs[:, oi]))
        # tournament on scalar (lower better); ranks all 0
        off = np.empty((P, M), dtype=int)
        pc, pm = 0.9 - 0.3 * g / C.GMAX, 0.05 + 0.15 * g / C.GMAX
        for i in range(0, P, 2):
            a = rng.integers(0, P, size=2)
            b = rng.integers(0, P, size=2)
            p1 = pop[a[np.argmin(s[a])]]
            p2 = pop[b[np.argmin(s[b])]]
            if rng.random() < pc:
                mask = rng.random(M) < 0.5
                c1 = np.where(mask, p1, p2)
                c2 = np.where(mask, p2, p1)
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
        comb = np.vstack([F, F2])
        comb_s = finite_F(comb)
        order = np.argsort(_scalar(normalize_minmax(comb_s[:, oi])))
        keep = order[:P]
        pop = np.vstack([pop, off])[keep]
        F = comb[keep]
        Fs = comb_s[keep]
        best_row = int(np.argmin(_scalar(normalize_minmax(Fs[:, oi]))))
        best_so_far = F[best_row].copy()
        if g % 10 == 0:
            front_history.append((g, best_so_far.reshape(1, -1)))
        g += 1

    # ---------------- Phase 2: simulated annealing ------------------------
    ga_lo = Fs[:, oi].min(axis=0)
    ga_hi = Fs[:, oi].max(axis=0)
    span = np.where(ga_hi - ga_lo > 1e-12, ga_hi - ga_lo, 1.0)

    def scalar_fixed(f_full: np.ndarray) -> float:
        fn = (finite_F(f_full)[oi] - ga_lo) / span
        return float(fn @ W_EQ)

    x = F[best_row].copy()          # current SA solution (full 4-obj)
    best = x.copy()
    s_cur = scalar_fixed(x)
    s_best = s_cur
    T = SA_T0
    sa_iters = 0
    while budget.can_afford(1):
        nb_assign = pop[best_row].copy()
        # neighbour: one swap of the current best assignment
        u, v = rng.choice(M, size=2, replace=False)
        nb_assign = nb_assign.copy()
        tmp = nb_assign[u]; nb_assign[u] = nb_assign[v]; nb_assign[v] = tmp
        budget.spend(1)
        evn = evaluate_population(nb_assign.reshape(1, -1), sc, cfg, h_cum)
        Fn_ = np.stack([evn["f1"], evn["f2"], evn["f3"], evn["f4"]], axis=1)[0]
        s_nb = scalar_fixed(Fn_)
        d = s_nb - s_cur
        if d <= 0.0 or rng.random() < float(np.exp(-d / T)):
            pop[best_row] = nb_assign
            F[best_row] = Fn_
            Fs[best_row] = finite_F(Fn_)
            x = Fn_
            s_cur = s_nb
            if s_nb < s_best:
                s_best = s_nb
                best = Fn_
        T *= SA_ALPHA
        sa_iters += 1
        if sa_iters % 10 == 0:
            front_history.append((g + sa_iters // 1000, best.reshape(1, -1)))

    wall = time.perf_counter() - start
    kidx = int(np.argmin(_scalar(normalize_minmax(finite_F(F)[:, oi]))))
    fronts_final = [np.asarray([kidx], dtype=int)]
    params = {"np": P, "budget": budget.limit, "obj_idx": OBJ_IDX,
              "ga_share": GA_SHARE, "ga_evals": budget.spent - sa_iters,
              "sa_iters": sa_iters, "sa_T0": SA_T0, "sa_alpha": SA_ALPHA,
              "weights": "equal-0.25", "evals": budget.spent}
    return AlgorithmResult(
        name=NAME, seed=seed, assign_final=pop, F_final=F,
        obj_idx=OBJ_IDX, fronts=fronts_final, knee_idx=kidx,
        best_assign=pop[kidx].copy(), best_F=F[kidx],
        evals=budget.spent, wall_s=wall,
        front_history=front_history, params=params)
