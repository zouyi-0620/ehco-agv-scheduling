"""E1 main comparison runner (SPEC.md sections 8, 9, 11).

Fair comparison protocol:
- 30 fixed seeds (1..30, written to results/seeds.json) shared by every
  algorithm (same Scenario instance per seed - tasks, AGV init, noise).
- Equal 20,000-eval budget per algorithm run.
- Per seed, the HV normalisation follows D5: min-max over the union of ALL
  algorithms' final fronts (in the algorithm's own objective subspace), then
  reference point (1,1,...,1).

Outputs (results/e1/):
  e1_solutions.csv   : every final-front solution of every algorithm-run
                       (seed, algo, row, f1..f4, knee_flag)
  e1_metrics.csv     : per algorithm-run summary (knee solution objectives,
                       HV, |F1|, evals, wall_s, delta-h fleet stats)
  e1_compare.csv     : per (algo, metric) mean +- std over the 30 seeds
  e1_stats.csv       : pairwise Wilcoxon + Holm + Cohen's d_z + 95% CI vs
                       AW-NSGA-II (D9)
  seeds.json         : the fixed seed list

Optional CLI: --seeds "1,2,3" --algo "AW-NSGA-II,MOEA/D" for quick runs.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np

from .. import constants as C
from ..algorithms import ORDER, REGISTRY, run_algorithm
from ..algorithms.common import AlgorithmResult, finite_F
from ..metrics import (ci95, cohen_dz, fast_non_dominated_sort, gini,
                       holm_adjust, hypervolume, paired_wilcoxon)
from ..objectives import build_plan, simulate_plan_health
from ..scenario import make_scenario

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "e1")
SEEDS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "seeds.json")
REF = np.ones(4)


def _load_or_write_seeds(path: str) -> list[int]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    seeds = list(C.SEEDS)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(seeds, f, indent=1)
    return seeds


def _fleet_health(res: AlgorithmResult, sc,
                  h0_coupling: float = 0.0) -> dict:
    """delta-h stats from a true-degradation replay of the reported solution."""
    plans = build_plan(res.best_assign, sc)
    sim = simulate_plan_health(plans, sc,
                               h0_coupling=h0_coupling)
    return {"delta_h_mean": sim["delta_h_mean"],
            "delta_h_max": float(sim["delta_h_max"]),
            "gini_delta_h": sim["gini"],
            "T_end_max": float(sim["T_end"].max()),
            "cycles_max": float(sim["cycles"].max())}


def _hv_in_subspace(F: np.ndarray, union: np.ndarray,
                    obj_idx: tuple[int, ...]) -> float:
    """D5 HV: min-max normalise the front by the union, ref (1,...,1)."""
    Fsub = F[:, list(obj_idx)]
    Usub = union[:, list(obj_idx)]
    lo = Usub.min(axis=0)
    hi = Usub.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    Fn = (Fsub - lo) / span
    return hypervolume(Fn, np.ones(len(obj_idx)))


def run_e1(seeds: list[int] | None = None,
           algos: list[str] | None = None,
           out_dir: str | None = None,
           verbose: bool = True) -> dict:
    seeds = seeds or _load_or_write_seeds(SEEDS_PATH)
    algos = algos or ORDER
    out_dir = out_dir or RESULTS
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # phase 1: run everything (parallelisable over seeds)
    # ------------------------------------------------------------------
    results: dict[tuple[str, int], AlgorithmResult] = {}
    scenarios: dict[int, object] = {}
    t_start = time.perf_counter()
    for s in seeds:
        sc = make_scenario(s)
        scenarios[s] = sc
        for name in algos:
            t0 = time.perf_counter()
            res = run_algorithm(name, sc, s)
            results[(name, s)] = res
            if verbose:
                print(f"  seed {s:3d} {name:<14} {time.perf_counter()-t0:5.2f}s "
                      f"|F1|={len(res.fronts[0]):3d} "
                      f"knee f1={res.best_F[0]:7.1f}s f2={res.best_F[1]:7.1f}",
                      flush=True)
    if verbose:
        print(f"[E1] {len(seeds)} seeds x {len(algos)} algos: "
              f"{time.perf_counter()-t_start:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # phase 2: D5 union-normalised HV per seed
    #   hv  : FULL 4-D space for every algorithm (cross-algorithm comparable;
    #         single-objective baselines honestly get a small point-HV)
    #   hv_sub : algorithm's own objective subspace (dimensionality-matched
    #         comparison, e.g. NSGA-II-2obj 2D HV vs 4-obj 2D projections)
    # ------------------------------------------------------------------
    hv_map: dict[tuple[str, int], float] = {}
    hv_sub_map: dict[tuple[str, int], float] = {}

    def _front4(res: AlgorithmResult) -> np.ndarray:
        idx = fast_non_dominated_sort(finite_F(res.F_final))[0]
        return res.F_final[idx]

    for s in seeds:
        union4 = np.vstack([_front4(results[(name, s)]) for name in algos])
        union_sub: dict[tuple[int, ...], list[np.ndarray]] = defaultdict(list)
        for name in algos:
            res = results[(name, s)]
            union_sub[res.obj_idx].append(res.F_final[res.fronts[0]])
        for name in algos:
            res = results[(name, s)]
            hv_map[(name, s)] = _hv_in_subspace(_front4(res), union4, (0, 1, 2, 3))
            U = np.vstack(union_sub[res.obj_idx])
            hv_sub_map[(name, s)] = _hv_in_subspace(
                res.F_final[res.fronts[0]], U, res.obj_idx)

    # ------------------------------------------------------------------
    # phase 3: persist
    # ------------------------------------------------------------------
    sol_rows, met_rows = [], []
    for (name, s), res in results.items():
        front = res.fronts[0]
        for r_i in front:
            sol_rows.append({
                "seed": s, "algo": name, "row": int(r_i),
                "f1": float(res.F_final[r_i][0]),
                "f2": float(res.F_final[r_i][1]),
                "f3": float(res.F_final[r_i][2]),
                "f4": float(res.F_final[r_i][3]),
                "knee": int(r_i == res.knee_idx),
            })
        fh = _fleet_health(res, scenarios[s])
        met_rows.append({
            "seed": s, "algo": name,
            "f1": float(res.best_F[0]), "f2": float(res.best_F[1]),
            "f3": float(res.best_F[2]), "f4": float(res.best_F[3]),
            "hv": hv_map[(name, s)], "hv_sub": hv_sub_map[(name, s)],
            "front_size": int(len(front)),
            "evals": res.evals, "wall_s": round(res.wall_s, 3),
            **fh,
        })
    _write_csv(os.path.join(out_dir, "e1_solutions.csv"), sol_rows)
    _write_csv(os.path.join(out_dir, "e1_metrics.csv"), met_rows)

    # ------------------------------------------------------------------
    # phase 4: aggregation + D9 statistics vs AW-NSGA-II
    # ------------------------------------------------------------------
    met = defaultdict(list)
    for m in met_rows:
        met[m["algo"]].append(m)
    agg_rows, stat_rows = [], []
    base = "AW-NSGA-II"
    for name in algos:
        rows = met[name]
        for k in ("f1", "f2", "f3", "f4", "hv", "hv_sub",
                  "delta_h_mean", "gini_delta_h"):
            vals = np.array([r[k] for r in rows], dtype=float)
            lo, hi = ci95(vals)
            agg_rows.append({"algo": name, "metric": k,
                             "mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
                             "ci95_lo": lo, "ci95_hi": hi})
        if name != base:
            a = np.array([r["hv"] for r in met[base]], dtype=float)
            b = np.array([r["hv"] for r in rows], dtype=float)
            p = paired_wilcoxon(b, a, alternative="less")   # b worse than a?
            stat_rows.append({
                "metric": "hv", "algo_a": base, "algo_b": name,
                "wilcoxon_p": p, "cohen_dz": cohen_dz(b, a),
                "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            })
    if stat_rows:
        pvals = np.array([r["wilcoxon_p"] for r in stat_rows])
        adj = holm_adjust(pvals)
        for r, pa in zip(stat_rows, adj):
            r["holm_p"] = pa
    _write_csv(os.path.join(out_dir, "e1_compare.csv"), agg_rows)
    _write_csv(os.path.join(out_dir, "e1_stats.csv"), stat_rows)

    return {"n_seeds": len(seeds), "n_algos": len(algos),
            "out_dir": out_dir, "solutions": len(sol_rows),
            "metrics": len(met_rows)}


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="comma list, e.g. 1,2,3")
    ap.add_argument("--algo", default=None, help="comma list of algorithm names")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    algos = args.algo.split(",") if args.algo else None
    run_e1(seeds=seeds, algos=algos)


if __name__ == "__main__":
    main()
