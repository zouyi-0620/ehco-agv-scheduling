"""E2 ablation study runner (SPEC.md section 9, E2).

Variants (each differs from Full AW-NSGA-II by exactly one component):
  Full            : AW-NSGA-II complete (4-obj, FLC weights, full health model,
                    improved A*); REUSED from E1 (results/e1) so that Table 3
                    and Table 1 share identical Full numbers.
  -health         : health objective removed (no f4, no maintenance cost, no
                    hard exclusion) -> optimises (f1, f2, f3) with FLC weights
                    projected onto the remaining objectives.
  -adaptW         : FLC replaced by fixed equal weights (0.25 each), 4-obj.
  -impAstar       : improved A* replaced by standard distance-only A*
                    (EvalConfig.use_improved_astar=False), 4-obj + FLC.
  -closedLoop     : FLC weights computed ONCE from the initial state and kept
                    fixed during the run (open-loop; no state feedback).
  hard-only       : full health model replaced by hard exclusion only
                    (AGVs with h_cum < 0.6 ineligible, no f4 / no maintenance
                    cost) -> optimises (f1, f2, f3) + FLC weights projected.
  GA-SA-multiSoH  : GA-SA baseline driven by the FULL multi-indicator SoH model
                    (EvalConfig() default: f4 + maintenance cost + h>=h_crit
                    hard constraint) instead of the battery-only h<0.6
                    exclusion; equal 0.25 scalar weights (D12).

Statistics: for each variant, paired Wilcoxon (by seed) vs Full on every
reported metric, Holm-Bonferroni adjusted across the variant comparisons;
Cohen's d_z and Student-t 95% CI on the mean.

Outputs (results/e2/): e2_solutions.csv, e2_metrics.csv, e2_compare.csv,
e2_stats.csv (same schema as e1_*).

Run from the project root:  python -m sim.experiments.e2_ablation
Optional CLI: --seeds "1,2,3" --variant "-health,hard-only" --reuse-e1 0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np

from .. import constants as C
from ..algorithms import aw_nsga2, ga_sa
from ..algorithms.common import (AlgorithmResult, EvalBudget, finite_F,
                                 flc_weight_provider, nsga2_core)
from ..metrics import (ci95, cohen_dz, fast_non_dominated_sort, gini,
                       holm_adjust, hypervolume, paired_wilcoxon)
from ..objectives import EvalConfig, build_plan, simulate_plan_health
from ..scenario import make_scenario
from .e1_main import RESULTS as E1_DIR, _fleet_health, _hv_in_subspace

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "e2")
SEEDS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "seeds.json")
REF = np.ones(4)

FULL_NAME = "Full"


# --------------------------------------------------------------------------
# variant runners
# --------------------------------------------------------------------------

def _fixed_weight_provider(sc, h_cum) -> np.ndarray:
    return np.array([0.25, 0.25, 0.25, 0.25])


class _OpenLoopFLC:
    """FLC computed once at run start; state feedback disabled (open loop)."""

    def __init__(self, sc):
        h0 = np.array([a.h0 for a in sc.agvs])
        self._w = np.asarray(flc_weight_provider(sc, h0), dtype=float)

    def __call__(self, sc, h_cum) -> np.ndarray:
        return self._w


def _run_health_ablated(sc, seed, cfg: EvalConfig) -> AlgorithmResult:
    """3-obj (f1,f2,f3) NSGA-II core with FLC weights projected on 3 targets."""
    return nsga2_core(
        sc, seed, obj_idx=(0, 1, 2), cfg=cfg, budget=EvalBudget(),
        name="-health", weight_fn=flc_weight_provider, use_crowding=True,
        params_extra={"weight_source": "FLC(3-obj)", "variant": "-health"})


def _run_fixed_w(sc, seed, cfg: EvalConfig) -> AlgorithmResult:
    return nsga2_core(
        sc, seed, obj_idx=(0, 1, 2, 3), cfg=cfg, budget=EvalBudget(),
        name="-adaptW", weight_fn=_fixed_weight_provider, use_crowding=True,
        params_extra={"weight_source": "fixed-0.25", "variant": "-adaptW"})


def _run_std_astar(sc, seed, cfg: EvalConfig) -> AlgorithmResult:
    return nsga2_core(
        sc, seed, obj_idx=(0, 1, 2, 3), cfg=cfg, budget=EvalBudget(),
        name="-impAstar", weight_fn=flc_weight_provider, use_crowding=True,
        params_extra={"weight_source": "FLC", "astar": "standard",
                      "variant": "-impAstar"})


def _run_open_loop(sc, seed, cfg: EvalConfig) -> AlgorithmResult:
    return nsga2_core(
        sc, seed, obj_idx=(0, 1, 2, 3), cfg=cfg, budget=EvalBudget(),
        name="-closedLoop", weight_fn=_OpenLoopFLC(sc), use_crowding=True,
        params_extra={"weight_source": "FLC-open-loop", "variant": "-closedLoop"})


def _run_hard_only(sc, seed, cfg: EvalConfig) -> AlgorithmResult:
    return nsga2_core(
        sc, seed, obj_idx=(0, 1, 2), cfg=cfg, budget=EvalBudget(),
        name="hard-only", weight_fn=flc_weight_provider, use_crowding=True,
        params_extra={"weight_source": "FLC(3-obj)", "hard": "h>=0.6",
                      "variant": "hard-only"})


def _run_ga_sa_multi(sc, seed, cfg: EvalConfig) -> AlgorithmResult:
    res = ga_sa.run(sc, seed, cfg=EvalConfig())
    res.name = "GA-SA-multiSoH"
    res.params["variant"] = "GA-SA-multiSoH"
    return res


# name -> (runner, EvalConfig)
VARIANTS: dict[str, tuple] = {
    "-health":     (_run_health_ablated, EvalConfig(use_health_objective=False,
                                                    use_maintenance_cost=False)),
    "-adaptW":     (_run_fixed_w, EvalConfig()),
    "-impAstar":   (_run_std_astar, EvalConfig(use_improved_astar=False)),
    "-closedLoop": (_run_open_loop, EvalConfig()),
    "hard-only":   (_run_hard_only, EvalConfig(hard_exclusion_only=True,
                                               use_health_objective=False,
                                               use_maintenance_cost=False)),
    "GA-SA-multiSoH": (_run_ga_sa_multi, EvalConfig()),
}
ORDER = ["Full"] + list(VARIANTS)          # canonical table order


# --------------------------------------------------------------------------
# E1 reuse: load AW-NSGA-II (Full) results
# --------------------------------------------------------------------------

def _load_e1_full(seeds: list[int]) -> tuple[dict[int, dict], dict[int, np.ndarray]]:
    """Return (metrics_by_seed, solutions_by_seed) for AW-NSGA-II from E1."""
    sol_path = os.path.join(E1_DIR, "e1_solutions.csv")
    met_path = os.path.join(E1_DIR, "e1_metrics.csv")
    if not (os.path.exists(sol_path) and os.path.exists(met_path)):
        raise FileNotFoundError(
            "E1 results missing - run `python -m sim.experiments.e1_main` first")
    met: dict[int, dict] = {}
    with open(met_path, encoding="utf-8") as f:
        cols = f.readline().strip().split(",")
        for line in f:
            r = dict(zip(cols, line.strip().split(",")))
            if r["algo"] != "AW-NSGA-II":
                continue
            s = int(r["seed"])
            met[s] = {k: float(r[k]) for k in
                      ("f1", "f2", "f3", "f4", "hv", "hv_sub",
                       "front_size", "evals", "wall_s", "delta_h_mean",
                       "delta_h_max", "gini_delta_h", "T_end_max", "cycles_max")}
    sol: dict[int, np.ndarray] = {}
    with open(sol_path, encoding="utf-8") as f:
        cols = f.readline().strip().split(",")
        for line in f:
            r = dict(zip(cols, line.strip().split(",")))
            if r["algo"] != "AW-NSGA-II":
                continue
            s = int(r["seed"])
            sol.setdefault(s, []).append(
                [float(r["f1"]), float(r["f2"]), float(r["f3"]), float(r["f4"])])
    sol = {s: np.asarray(v) for s, v in sol.items()}
    missing = [s for s in seeds if s not in met]
    if missing:
        raise FileNotFoundError(
            f"E1 AW-NSGA-II results incomplete for seeds {missing}")
    return met, sol


# --------------------------------------------------------------------------
# main runner
# --------------------------------------------------------------------------

def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)")


def run_e2(seeds: list[int] | None = None,
           variants: list[str] | None = None,
           reuse_e1: bool = True,
           verbose: bool = True) -> dict:
    seeds = seeds or list(C.SEEDS)
    variants = variants or list(VARIANTS)
    if FULL_NAME in variants:
        raise ValueError("'Full' is reused from E1, do not list it in --variant")

    # ---- Full from E1 ---------------------------------------------------
    full_met, full_sol = _load_e1_full(seeds)
    full_metrics: dict[int, dict] = {
        s: {"variant": FULL_NAME, "seed": s, **full_met[s]} for s in seeds}

    # -closedLoop is bit-identical to Full in the STATIC scenario: h_cum is
    # fixed at its initial value during static evaluation, so the FLC input
    # (rho, gamma, h_bar) is constant across generations and the open-loop
    # weights equal the closed-loop weights at every generation. We therefore
    # reuse the Full data (no wasted compute) and report the equality
    # explicitly - the closed-loop contribution is genuinely tested by E4
    # (dynamic scenarios) where h_cum evolves. See SPEC.md D1/E2.
    closed_loop_present = "-closedLoop" in variants
    variants_run = [v for v in variants if v != "-closedLoop"]
    order = [FULL_NAME] + variants          # table order = Full + requested

    # ---- run the remaining variants -------------------------------------
    results: dict[tuple[str, int], AlgorithmResult] = {}
    scenarios: dict[int, object] = {}
    t_start = time.perf_counter()
    for s in seeds:
        sc = make_scenario(s)
        scenarios[s] = sc
        for name in variants_run:
            runner, cfg = VARIANTS[name]
            t0 = time.perf_counter()
            res = runner(sc, s, cfg)
            results[(name, s)] = res
            if verbose:
                print(f"  seed {s:3d} {name:<14} {time.perf_counter()-t0:5.2f}s "
                      f"|F1|={len(res.fronts[0]):3d} "
                      f"knee f1={res.best_F[0]:7.1f}s f2={res.best_F[1]:7.1f} "
                      f"f4={res.best_F[3]:.4f}", flush=True)
    if verbose:
        print(f"[E2] {len(seeds)} seeds x {len(variants_run)} variants run "
              f"({'+Full reuse for -closedLoop'} if closed_loop_present else ''): "
              f"{time.perf_counter()-t_start:.1f}s", flush=True)

    # ---- D5 union-normalised HV per seed (per-run union over ALL variants)
    def _front4(name, s) -> np.ndarray:
        if name in (FULL_NAME, "-closedLoop"):
            return full_sol[s]
        res = results[(name, s)]
        idx = fast_non_dominated_sort(finite_F(res.F_final))[0]
        return res.F_final[idx]

    # per-seed objective subspace for each variant (D5 hv_sub, E2-internal
    # union normalisation so Full and the variants share one consistent basis)
    obj_idx_of = {FULL_NAME: (0, 1, 2, 3), "-closedLoop": (0, 1, 2, 3)}
    for name in variants:
        if name in ("GA-SA-multiSoH", "-closedLoop"):
            obj_idx_of[name] = (0, 1, 2, 3)
        else:
            obj_idx_of[name] = results[(name, seeds[0])].obj_idx

    sol_rows, met_rows = [], []
    for s in seeds:
        union4 = np.vstack([_front4(name, s) for name in order])
        # hv_sub unions per objective subspace
        sub_union: dict[tuple[int, ...], list[np.ndarray]] = defaultdict(list)
        for name in order:
            if name in (FULL_NAME, "-closedLoop"):
                sub_union[obj_idx_of[name]].append(full_sol[s])
            else:
                res = results[(name, s)]
                sub_union[obj_idx_of[name]].append(res.F_final[res.fronts[0]])
        for name in order:
            if name in (FULL_NAME, "-closedLoop"):
                front = np.arange(len(full_sol[s]))
                F = full_sol[s]
                knee_row = None           # knee flagged in e1_solutions
                m = full_metrics[s]
            else:
                res = results[(name, s)]
                F = res.F_final
                front = res.fronts[0]
                knee_row = res.knee_idx
            for r_i in front:
                sol_rows.append({
                    "seed": s, "variant": name, "row": int(r_i),
                    "f1": float(F[r_i][0]), "f2": float(F[r_i][1]),
                    "f3": float(F[r_i][2]), "f4": float(F[r_i][3]),
                    "knee": int(knee_row is not None and r_i == knee_row),
                })
            oi = obj_idx_of[name]
            hv = _hv_in_subspace(F[front], union4, (0, 1, 2, 3))
            U = np.vstack(sub_union[oi])
            hv_sub = _hv_in_subspace(F[front], U, oi)
            if name in (FULL_NAME, "-closedLoop"):
                met_rows.append({"seed": s, "variant": name,
                                 "f1": m["f1"], "f2": m["f2"], "f3": m["f3"],
                                 "f4": m["f4"], "hv": hv, "hv_sub": hv_sub,
                                 "front_size": int(m["front_size"]),
                                 "evals": int(m["evals"]),
                                 "wall_s": m["wall_s"],
                                 "delta_h_mean": m["delta_h_mean"],
                                 "delta_h_max": m["delta_h_max"],
                                 "gini_delta_h": m["gini_delta_h"],
                                 "T_end_max": m["T_end_max"],
                                 "cycles_max": m["cycles_max"]})
            else:
                fh = _fleet_health(res, scenarios[s])
                met_rows.append({"seed": s, "variant": name,
                                 "f1": float(res.best_F[0]),
                                 "f2": float(res.best_F[1]),
                                 "f3": float(res.best_F[2]),
                                 "f4": float(res.best_F[3]),
                                 "hv": hv, "hv_sub": hv_sub,
                                 "front_size": int(len(front)),
                                 "evals": res.evals,
                                 "wall_s": res.wall_s,
                                 **fh})
    os.makedirs(RESULTS, exist_ok=True)
    _write_csv(os.path.join(RESULTS, "e2_solutions.csv"), sol_rows)
    _write_csv(os.path.join(RESULTS, "e2_metrics.csv"), met_rows)

    # ---- aggregation + stats vs Full ------------------------------------
    met = defaultdict(list)
    for m in met_rows:
        met[m["variant"]].append(m)
    agg_rows, stat_rows = [], []
    for name in order:
        rows = met[name]
        for k in ("f1", "f2", "f3", "f4", "hv", "delta_h_mean", "gini_delta_h"):
            vals = np.array([r[k] for r in rows], dtype=float)
            lo, hi = ci95(vals)
            agg_rows.append({"variant": name, "metric": k,
                             "mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
                             "ci95_lo": lo, "ci95_hi": hi})
        if name != FULL_NAME:
            a = np.array([r["hv"] for r in met[FULL_NAME]], dtype=float)
            b = np.array([r["hv"] for r in rows], dtype=float)
            p = paired_wilcoxon(b, a, alternative="less")
            stat_rows.append({
                "metric": "hv", "variant_a": FULL_NAME, "variant_b": name,
                "wilcoxon_p": p, "cohen_dz": cohen_dz(b, a),
                "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            })
    if stat_rows:
        pvals = np.array([r["wilcoxon_p"] for r in stat_rows])
        adj = holm_adjust(pvals)
        for r, pa in zip(stat_rows, adj):
            r["holm_p"] = pa
    _write_csv(os.path.join(RESULTS, "e2_compare.csv"), agg_rows)
    _write_csv(os.path.join(RESULTS, "e2_stats.csv"), stat_rows)

    return {"n_seeds": len(seeds), "n_variants": len(order),
            "out_dir": RESULTS, "solutions": len(sol_rows),
            "metrics": len(met_rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="comma list, e.g. 1,2,3")
    ap.add_argument("--variant", default=None, help="comma list of variant names")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    variants = args.variant.split(",") if args.variant else None
    run_e2(seeds=seeds, variants=variants)


if __name__ == "__main__":
    main()
