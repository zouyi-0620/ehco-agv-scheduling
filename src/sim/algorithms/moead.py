"""MOEA/D baseline (SPEC.md section 8): Tchebycheff decomposition, 100 uniform
weight vectors, neighbourhood T=20, equal 20,000-eval budget.

Decision points (SPEC D12):
- No integer Das-Dennis division gives exactly 100 vectors in 4 objectives, so
  the 100 weight vectors are Dirichlet(1,1,1,1) draws from a FIXED generator
  seed (config constant, identical across scenario seeds for fairness).
- Offspring: uniform column-wise crossover of two neighbourhood solutions
  (pc = 1, MOEA/D style) + one swap (prob 0.5) + per-gene reassign pm = 1/M.
- Offspring are batch-evaluated per generation (evaluation is stateless per
  individual), then the sequential neighbourhood update is applied - results
  are identical to evaluate-on-the-fly but ~10x faster.
"""
from __future__ import annotations

import time

import numpy as np

from .. import constants as C
from ..metrics import dirichlet_weights, fast_non_dominated_sort
from ..objectives import EvalConfig, evaluate_population, knee_point_selection
from ..scenario import Scenario
from .common import (AlgorithmResult, EvalBudget, eligible_mask, finite_F,
                     init_population, mutate, repair, uniform_col_crossover_pair)

NAME = "MOEA/D"
OBJ_IDX = (0, 1, 2, 3)
WV_SEED = 20260901                 # fixed config seed for the weight vectors


def _tcheby(f: np.ndarray, w: np.ndarray, z: np.ndarray) -> float:
    return float(np.max(w * (f - z)))


def run(sc: Scenario, seed: int, cfg: EvalConfig | None = None,
        budget: EvalBudget | None = None,
        tracker=None) -> AlgorithmResult:
    # `tracker`: optional callable(g, elapsed_s, F_of_pop) invoked at every
    # generation end for anytime / best-so-far studies (B2).  Default None =>
    # behaviour bit-identical to the pre-tracker code path.  Note F is mutated
    # in place across generations in MOEA/D, so tracker receives a copy.
    cfg = cfg or EvalConfig()
    budget = budget or EvalBudget()
    rng = np.random.default_rng(seed)
    N = len(sc.agvs)
    M = len(sc.tasks)
    P = C.MOEAD_N_WV
    T = C.MOEAD_T
    oi = list(OBJ_IDX)
    h_cum = np.array([a.h0 for a in sc.agvs])
    elig = eligible_mask(h_cum, cfg)

    # ---- weight vectors + neighbourhood --------------------------------
    W = dirichlet_weights(P, len(OBJ_IDX), WV_SEED)          # (P, 4), rows sum 1
    D = np.linalg.norm(W[:, None, :] - W[None, :, :], axis=2)
    neigh = np.argsort(D, axis=1)[:, :T]                     # (P, T)
    # ensure self is included (nearest vector is itself, distances >= 0)
    B = np.where(neigh == np.arange(P)[:, None], neigh,
                 neigh)                                      # keep as-is

    # ---- init ----------------------------------------------------------
    pop = init_population(P, M, elig, rng)
    budget.spend(P)
    ev = evaluate_population(pop, sc, cfg, h_cum)
    F = np.stack([ev["f1"], ev["f2"], ev["f3"], ev["f4"]], axis=1)
    Fs = finite_F(F)
    z_star = Fs[:, oi].min(axis=0)

    front_history: list[tuple[int, np.ndarray]] = []
    start = time.perf_counter()
    g = 0
    while budget.can_afford(P):
        # ---- generate offspring for every subproblem ---------------------
        off = np.empty((P, M), dtype=int)
        pm = 1.0 / M
        for i in range(P):
            a, b = rng.choice(B[i], size=2, replace=False)
            c1, _ = uniform_col_crossover_pair(pop[a], pop[b], rng)
            if rng.random() < 0.5 and M >= 2:                # one swap
                u, v = rng.choice(M, size=2, replace=False)
                tmp = c1[u]; c1[u] = c1[v]; c1[v] = tmp
            hit = rng.random(M) < pm
            if hit.any():
                e_idx = np.where(elig)[0]
                c1[hit] = e_idx[rng.integers(0, len(e_idx), size=int(hit.sum()))]
            off[i] = c1
        repair(off, elig, rng)

        # ---- batch evaluate + sequential neighbourhood update -------------
        budget.spend(P)
        ev2 = evaluate_population(off, sc, cfg, h_cum)
        F2 = np.stack([ev2["f1"], ev2["f2"], ev2["f3"], ev2["f4"]], axis=1)
        F2s = finite_F(F2)
        for i in range(P):
            z_star = np.minimum(z_star, F2s[i][oi])
            g_new = _tcheby(F2s[i][oi], W[i], z_star)
            for j in B[i]:
                if _tcheby(F2s[i][oi], W[j], z_star) <= _tcheby(Fs[j][oi], W[j], z_star):
                    pop[j] = off[i]
                    Fs[j] = F2s[i]
                    F[j] = F2[i]

        if g % 10 == 0:
            f_now = fast_non_dominated_sort(Fs[:, oi])
            front_history.append((g, F[f_now[0]].copy()))
        if tracker is not None:
            tracker(g, time.perf_counter() - start, F.copy())
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

    params = {"np": P, "budget": budget.limit, "obj_idx": OBJ_IDX,
              "T": T, "weight_vectors": P,
              "weight_generation": f"dirichlet-seed{WV_SEED}",
              "decomposition": "Tchebycheff", "gens_actual": g,
              "evals": budget.spent}
    return AlgorithmResult(
        name=NAME, seed=seed, assign_final=pop, F_final=F,
        obj_idx=OBJ_IDX, fronts=fronts_final, knee_idx=kidx,
        best_assign=pop[kidx].copy(), best_F=F[kidx],
        evals=budget.spent, wall_s=wall,
        front_history=front_history, params=params)
