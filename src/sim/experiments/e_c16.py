"""C16: health-dimension activation experiment (reviewer C-group, R1-M2/R3-M4/R2-M3).

Motivation
----------
In the published static experiments (E1/E2) the initial fleet health is drawn
from U[0.7, 1.0], so every AGV starts above h_safe = 0.6 and the piecewise
health penalty f4 is identically zero: the "continuous health penalty" arm of
the model is never exercised.  The ablation Full == hard-only (bit-identical)
is a direct consequence.  C16 activates the health dimension:

  * 5 of 10 AGVs start with h0 ~ U[0.3, 0.6)  (below h_safe, above h_crit)
  * 5 of 10 AGVs start with h0 ~ U[0.7, 1.0]  (healthy, as published)

Tasks, AGV positions and the seed stream are IDENTICAL to E1 (make_scenario
is called with the published parameters; only the h0 values are overwritten
afterwards, drawn from an independent seeded RNG so the protocol is exact and
reproducible).

Arms (same 20,000-eval budget, same NSGA-II core as E2):
  full      : AW-NSGA-II complete model (f1..f4 with f4 ACTIVE, maintenance
              cost, hard exclusion only at h < h_crit = 0.2)
  hard      : hard exclusion at h < h_safe = 0.6, no f4 / no c_m
              (E2 "hard-only" runner -> the 5 weak AGVs are now genuinely
              excluded, unlike in E2 where the arm was vacuous)
  nohealth  : no health objective at all (E2 "-health" runner)
  moead     : MOEA/D with the full 4-objective model (f4 active) -- the EMO
              comparison under an activated health dimension (R2-M3)

Metrics
-------
Per-run decision metrics are evaluated under a COMMON full-model evaluation
(the published objective definition) so the arms are comparable:
  f1 (makespan), f2_full (cost incl. c_m*(1-h)), f4_full (health penalty of
  the chosen plan), share_low (fraction of tasks assigned to weak AGVs).
True degradation is measured by the plan replay (simulate_plan_health):
  delta_h_mean/max, gini, h_end_min, delta_h on the weak vs healthy group.

Statistics: paired two-sided Wilcoxon by seed, Holm-adjusted across the whole
test family; Cohen's d_z.

Outputs (results/e_c16/):
  c16_metrics.csv  per (arm, seed)
  c16_compare.csv  mean +- std per (arm, metric)
  c16_stats.csv    pairwise two-sided Wilcoxon vs `full` (+ moead vs full HV)
  c16_meta.json    protocol dump

Run from the project root:  python -m sim.experiments.e_c16
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np

from .. import constants as C
from ..algorithms import moead
from ..algorithms.common import (EvalBudget, flc_weight_provider, nsga2_core)
from ..metrics import ci95, cohen_dz, fast_non_dominated_sort, holm_adjust, \
    paired_wilcoxon
from ..objectives import (EvalConfig, build_plan, evaluate_population,
                          simulate_plan_health)
from ..scenario import make_scenario

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "e_c16")
SEEDS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "seeds.json")

N_LOW = 5                    # weak AGVs per fleet
LOW_LO, LOW_HI = 0.30, 0.60  # below h_safe, above h_crit -> f4 active
HIGH_LO, HIGH_HI = 0.70, 1.00
HEALTH_RNG_OFFSET = 90210    # independent stream for the h0 override

FULL_EVAL = EvalConfig()     # common evaluation model (published definition)


# --------------------------------------------------------------------------
# scenario construction
# --------------------------------------------------------------------------

def make_c16_scenario(seed: int):
    """E1 scenario (identical tasks/positions/seed stream) with 5 weak AGVs.

    Returns (scenario, low_ids) where low_ids is the set of AGV indices whose
    h0 was drawn from U[0.3, 0.6)."""
    sc = make_scenario(seed)
    rng = np.random.default_rng(HEALTH_RNG_OFFSET + seed)
    order = rng.permutation(len(sc.agvs))
    low_ids = set(int(i) for i in order[:N_LOW])
    for i in order[:N_LOW]:
        sc.agvs[int(i)].h0 = float(rng.uniform(LOW_LO, LOW_HI))
    for i in order[N_LOW:]:
        sc.agvs[int(i)].h0 = float(rng.uniform(HIGH_LO, HIGH_HI))
    return sc, low_ids


# --------------------------------------------------------------------------
# arm runners (mirror E2 exactly; equal 20,000-eval budgets)
# --------------------------------------------------------------------------

def _run_full(sc, seed):
    # AW-NSGA-II complete: FLC weights steer, crowding preserves diversity
    # (identical call shape to algorithms/aw_nsga2.run, spelled out here with
    #  the explicit name so the arm table is self-documenting)
    return nsga2_core(sc, seed, obj_idx=(0, 1, 2, 3), cfg=EvalConfig(),
                      budget=EvalBudget(), name="full",
                      weight_fn=flc_weight_provider, use_crowding=True)


def _run_hard(sc, seed):
    # E2 "hard-only": exclusion at h < 0.6, no f4 / no maintenance cost
    return nsga2_core(sc, seed, obj_idx=(0, 1, 2),
                      cfg=EvalConfig(hard_exclusion_only=True,
                                     use_health_objective=False,
                                     use_maintenance_cost=False),
                      budget=EvalBudget(), name="hard",
                      weight_fn=flc_weight_provider, use_crowding=True,
                      params_extra={"hard": "h>=0.6", "variant": "hard"})


def _run_nohealth(sc, seed):
    # E2 "-health": no health objective at all
    return nsga2_core(sc, seed, obj_idx=(0, 1, 2),
                      cfg=EvalConfig(use_health_objective=False,
                                     use_maintenance_cost=False),
                      budget=EvalBudget(), name="nohealth",
                      weight_fn=flc_weight_provider, use_crowding=True,
                      params_extra={"variant": "nohealth"})


def _run_moead(sc, seed):
    return moead.run(sc, seed, cfg=EvalConfig(), budget=EvalBudget())


ARMS = {
    "full": _run_full,
    "hard": _run_hard,
    "nohealth": _run_nohealth,
    "moead": _run_moead,
}
ORDER = ["full", "hard", "nohealth", "moead"]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def _norm_cols(X: np.ndarray) -> np.ndarray:
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (X - lo) / span


def arm_metrics(res, sc, low_ids: set[int]) -> dict:
    """Common-model evaluation + true-degradation replay of the chosen plan.

    Three decision views are recorded because the published knee rule (SPEC
    D11) can land on a front extreme when the f4 dimension is active:
      * knee    : the published decision rule (primary, protocol-consistent)
      * min     : front extremes (f1_min = best makespan the arm's search
                  found; f4_full_min = best health penalty on the front)
      * eqw     : the front point minimising the equal-weight normalised
                  scalar under the COMMON full-model evaluation -- a single
                  decision rule applied identically to every arm
    """
    assign = res.best_assign.reshape(1, -1)
    ev = evaluate_population(assign, sc, FULL_EVAL)
    plans = build_plan(res.best_assign, sc)
    sim = simulate_plan_health(plans, sc)
    low = np.array([i in low_ids for i in range(len(sc.agvs))])
    dh = sim["delta_h"]
    share_low = float(low[res.best_assign].mean())

    # ---- whole-front common-model evaluation --------------------------
    front_idx = res.fronts[0]
    Ff = res.assign_final[front_idx]
    evf = evaluate_population(Ff, sc, FULL_EVAL)
    F_full = np.stack([evf["f1"], evf["f2"], evf["f3"], evf["f4"]], axis=1)
    f1_min = float(F_full[:, 0].min())
    f4_min = float(F_full[:, 3].min())
    eqw = _norm_cols(F_full).mean(axis=1)
    k_eqw = int(np.argmin(eqw))
    f1_eqw, f2_eqw, f4_eqw = (float(F_full[k_eqw, 0]), float(F_full[k_eqw, 1]),
                              float(F_full[k_eqw, 3]))

    return {
        # own-model reported objectives (knee solution)
        "f1": float(res.best_F[0]),
        "f2_own": float(res.best_F[1]),
        "f4_own": float(res.best_F[3]),
        # common full-model evaluation of the chosen plan (knee)
        "f2_full": float(ev["f2"][0]),
        "f4_full": float(ev["f4"][0]),
        # front extremes under the common model
        "f1_min": f1_min,
        "f4_full_min": f4_min,
        # equal-weight balanced decision under the common model
        "f1_eqw": f1_eqw, "f2_full_eqw": f2_eqw, "f4_full_eqw": f4_eqw,
        # decision structure
        "share_low": share_low,
        "n_tasks_low": int(low[res.best_assign].sum()),
        # true degradation replay (NOTE: in the static replay h0 does not
        # enter the degradation dynamics, so these measure realised fleet
        # dispersion, not protection of weak AGVs -- see experiment report)
        "delta_h_mean": sim["delta_h_mean"],
        "delta_h_max": float(sim["delta_h_max"]),
        "gini_delta_h": sim["gini"],
        "h_end_min": float(sim["h_end"].min()),
        "h_end_min_lowgrp": float(sim["h_end"][low].min()),
        "delta_h_low_mean": float(dh[low].mean()),
        "delta_h_high_mean": float(dh[~low].mean()),
        "T_end_max": float(sim["T_end"].max()),
        # bookkeeping
        "front_size": int(len(front_idx)),
        "evals": res.evals, "wall_s": round(res.wall_s, 3),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def _write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)", flush=True)


def run_c16(seeds: list[int] | None = None, arms: list[str] | None = None,
            verbose: bool = True) -> dict:
    if seeds is None:
        if os.path.exists(SEEDS_PATH):
            with open(SEEDS_PATH, encoding="utf-8") as f:
                seeds = json.load(f)
        else:
            seeds = list(C.SEEDS)
    arms = arms or ORDER
    os.makedirs(RESULTS, exist_ok=True)

    t0 = time.perf_counter()
    met_rows, low_map, hv_pairs = [], {}, {}
    for s in seeds:
        sc, low_ids = make_c16_scenario(s)
        low_map[s] = sorted(low_ids)
        for name in arms:
            res = ARMS[name](sc, s)
            m = arm_metrics(res, sc, low_ids)
            m.update({"arm": name, "seed": s})
            met_rows.append(m)
            if name in ("full", "moead"):
                # union-normalised 4-obj HV, full vs moead only (both 4-obj)
                hv_pairs.setdefault(s, {})[name] = res
            if verbose:
                print(f"  seed {s:3d} {name:<9} {res.wall_s:5.2f}s "
                      f"f1={m['f1']:7.1f}s f2_full={m['f2_full']:7.1f} "
                      f"f4_full={m['f4_full']:8.2f} share_low={m['share_low']:.2f} "
                      f"h_end_min={m['h_end_min']:.3f}", flush=True)

    # HV (full vs moead, union-normalised 4-D, D5-style within C16)
    hv_map: dict[tuple[str, int], float] = {}
    for s in seeds:
        if s not in hv_pairs or len(hv_pairs[s]) < 2:
            continue
        fronts = {}
        for name, res in hv_pairs[s].items():
            idx = fast_non_dominated_sort(
                res.F_final[np.isfinite(res.F_final).all(axis=1)])[0]
            fronts[name] = res.F_final[idx]
        union = np.vstack(list(fronts.values()))
        lo, hi = union.min(axis=0), union.max(axis=0)
        span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        for name, F in fronts.items():
            Fn = (F - lo) / span
            Fn = np.minimum(Fn, 1.0)
            from ..metrics import hypervolume
            hv_map[(name, s)] = hypervolume(Fn, np.ones(4))
    for r in met_rows:
        r["hv"] = hv_map.get((r["arm"], r["seed"]), "")

    _write_csv(os.path.join(RESULTS, "c16_metrics.csv"), met_rows)

    # ---- aggregation ------------------------------------------------------
    by = defaultdict(list)
    for r in met_rows:
        by[r["arm"]].append(r)
    agg_rows = []
    METRICS = ["f1", "f2_full", "f4_full", "f1_min", "f4_full_min",
               "f1_eqw", "f2_full_eqw", "f4_full_eqw", "share_low",
               "delta_h_mean", "delta_h_max", "gini_delta_h", "h_end_min",
               "h_end_min_lowgrp", "delta_h_low_mean", "delta_h_high_mean",
               "T_end_max", "hv"]
    for name in ORDER:
        if name not in by:
            continue
        rows = by[name]
        for k in METRICS:
            vals = np.array([float(r[k]) for r in rows if r[k] != ""],
                            dtype=float)
            if len(vals) == 0:
                continue
            lo95, hi95 = ci95(vals)
            agg_rows.append({"arm": name, "metric": k,
                             "mean": float(vals.mean()),
                             "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                             "ci95_lo": lo95, "ci95_hi": hi95, "n": len(vals)})
    _write_csv(os.path.join(RESULTS, "c16_compare.csv"), agg_rows)

    # ---- paired statistics vs `full` --------------------------------------
    stat_rows = []
    base = "full"
    TESTS = [("f1",), ("f2_full",), ("f4_full",), ("f1_min",),
             ("f1_eqw",), ("f2_full_eqw",), ("f4_full_eqw",),
             ("share_low",), ("gini_delta_h",), ("hv",)]
    for name in ORDER:
        if name == base or name not in by:
            continue
        for (k,) in TESTS:
            a = np.array([float(r[k]) for r in by[base] if r[k] != ""],
                         dtype=float)
            b = np.array([float(r[k]) for r in by[name] if r[k] != ""],
                         dtype=float)
            if len(a) == 0 or len(a) != len(b):
                continue
            p = paired_wilcoxon(b, a, alternative="two-sided")
            stat_rows.append({"metric": k, "arm_a": base, "arm_b": name,
                              "wilcoxon_p": p, "holm_p": "",
                              "cohen_dz": cohen_dz(b, a),
                              "mean_a": float(a.mean()),
                              "mean_b": float(b.mean()), "n": len(a)})
    if stat_rows:
        adj = holm_adjust(np.array([r["wilcoxon_p"] for r in stat_rows]))
        for r, pa in zip(stat_rows, adj):
            r["holm_p"] = float(pa)
    _write_csv(os.path.join(RESULTS, "c16_stats.csv"), stat_rows)

    meta = {
        "seeds": seeds, "arms": arms, "n_low": N_LOW,
        "low_range": [LOW_LO, LOW_HI], "high_range": [HIGH_LO, HIGH_HI],
        "health_rng_offset": HEALTH_RNG_OFFSET,
        "h_safe": C.H_SAFE, "h_crit": C.H_CRIT,
        "budget": 20000, "low_ids_per_seed": {str(s): low_map[s] for s in seeds},
        "stats": "paired two-sided Wilcoxon vs full, Holm across the family",
    }
    with open(os.path.join(RESULTS, "c16_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    if verbose:
        print(f"[C16] {len(seeds)} seeds x {len(arms)} arms: "
              f"{time.perf_counter()-t0:.1f}s", flush=True)
    return {"n_seeds": len(seeds), "n_arms": len(arms), "out_dir": RESULTS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="comma list, e.g. 1,2,3")
    ap.add_argument("--arm", default=None, help="comma list of arm names")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    arms = args.arm.split(",") if args.arm else None
    run_c16(seeds=seeds, arms=arms)


if __name__ == "__main__":
    main()
