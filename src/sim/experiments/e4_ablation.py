"""E4 dynamic ablation: isolate the FLC and adaptive-weight contributions.

Reviewer consensus blocker (R1-M2 / R2-M1 / R3-M4): the dynamic scenario only
compared the closed loop (EHCO) against an open-loop A*-GA baseline, so the
independent contributions of the FLC weight adaptation and of the
adaptive-weight initial plan were never isolated.  This script closes that
gap with two ablation variants run on the SAME 30 seeds x 5 instances:

  EHCO  = AW-NSGA-II initial plan + FLC-weighted replanning  (e4_raw.csv)
  -FLC  = AW-NSGA-II initial plan + fixed-weight NSGA-II replanning  (new)
  -AW   = fixed-weight NSGA-II-4obj initial plan + FLC replanning    (new)
  A*-GA = A*-GA initial plan + no replanning (open loop)     (e4_raw.csv)

Interpretation:
  Full vs -FLC  -> contribution of the FLC weight adaptation in replanning
  Full vs -AW   -> contribution of the adaptive-weight initial plan
  Full vs A*-GA -> total closed-loop benefit (already published)

The replanning budget, event instances and RNG streams are identical to the
published E4 protocol; only the weight provider (FLC on/off) or the initial
plan (AW vs fixed-weight NSGA-II-4obj, same 20,000-eval budget) changes.
All numpy operations must run on the managed venv (numpy 2.5.1) so that the
initial plans reproduce e1 / e4 exactly.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .. import constants as C
from ..algorithms import run_algorithm
from ..algorithms.common import EvalBudget, nsga2_core
from ..scenario import make_scenario
from .e4_dynamic import (DynamicSimulator, SEEDS, RESULTS, aggregate,
                         make_instances)

NEW_STRATEGIES = ("noFLC", "noAW")
PUB_STRATEGIES = ("EHCO", "A*-GA")


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)")


def run_ablation_scenario(kind: str, seed: int, inst: dict, strategy: str,
                          plans: dict) -> dict:
    """One (seed, instance, ablation-strategy) run; mirrors run_scenario in
    e4_dynamic.py so the row schema is identical."""
    sc = plans["scenarios"][seed]
    if strategy == "noFLC":
        assign, use_flc, closed_loop = plans["aw"][seed], False, True
    elif strategy == "noAW":
        assign, use_flc, closed_loop = plans["nsga2"][seed], True, True
    else:
        raise ValueError(f"unexpected ablation strategy {strategy}")
    ev = dict(inst)
    ev["rng"] = np.random.default_rng(seed * 1000 + 777)

    sim = DynamicSimulator(sc, assign, use_flc=use_flc)
    base = sim.run(event=None, closed_loop=False, ts_out=False)
    out = sim.run(event=ev, closed_loop=closed_loop, ts_out=False)

    f1_base = base["makespan"]
    f1_ev = out["makespan"]
    deg = (f1_ev - f1_base) / f1_base * 100.0 if f1_base > 0 else float("nan")
    row = {
        "scenario": kind, "strategy": strategy, "seed": seed,
        "instance": inst.get("_idx", 0),
        "t_ev": ev.get("t_ev") if ev.get("kind") != "fault" else ev.get("t_inj"),
        "f1_base": f1_base, "f1_event": f1_ev, "deg_pct": deg,
        "timeout": int(out["timeout"]),
        "response_ms": out["replan_walls_ms"][0] if out["replan_walls_ms"] else "",
        "replan_evals": out["replan_info"][0]["evals"] if out["replan_info"] else "",
        "warn_t": out["warn_t"], "warn_agv": out["warn_agv"],
        "n_warned": out["n_warned"], "n_agv": C.N_AGV,
        "event_fired": int(out["event_fired"]),
        "replan_applied": int(bool(out["replan_info"])
                              and not out["replan_info"][0].get("skipped", False)
                              and out["replan_info"][0].get("applied", True)),
        "maint_enter": out["maint_enter"] if kind == "fault" else "",
    }
    if kind == "fault":
        row["fault_agv"] = out["fault_agv"]
        row["fault_type"] = ev["fault_type"]
        row["fault_fallback"] = out["fault_fallback"]
        row["detected"] = int(out["warn_agv"] == out["fault_agv"])
        row["false_pos"] = int(sum(1 for i in out["warned_indices"]
                                   if i != out["fault_agv"]))
    return row


def load_published(seeds: list[int], scenarios: list[str],
                   e4_raw_path: str) -> list[dict]:
    """Reuse the EHCO / A*-GA rows of the published e4_raw.csv so the
    ablation is paired (same seed, same instance) with the published rows."""
    out = []
    with open(e4_raw_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            d = dict(zip(header, parts))
            if (d["strategy"] in PUB_STRATEGIES
                    and int(d["seed"]) in seeds
                    and d["scenario"] in scenarios):
                for k in ("f1_base", "f1_event", "deg_pct", "response_ms",
                          "replan_evals", "warn_t", "t_ev"):
                    if d[k] == "":
                        d[k] = None
                    else:
                        try:
                            d[k] = float(d[k])
                        except ValueError:
                            pass
                for k in ("timeout", "n_warned", "n_agv", "event_fired",
                          "replan_applied"):
                    d[k] = int(d[k])
                if d["scenario"] == "fault":
                    for k in ("fault_agv", "fault_type", "fault_fallback",
                              "detected", "false_pos"):
                        if k in d:
                            d[k] = int(d[k]) if d[k] != "" else None
                out.append(d)
    return out


def run_e4_ablation(seeds: list[int] | None = None,
                    scenarios: list[str] | None = None,
                    out_dir: str | None = None,
                    verbose: bool = True) -> dict:
    seeds = seeds or SEEDS
    scenarios = scenarios or ["congestion", "fault", "urgent"]
    out_dir = out_dir or RESULTS
    os.makedirs(out_dir, exist_ok=True)

    t_start = time.perf_counter()
    # ---- phase 0: initial plans ------------------------------------------
    if verbose:
        print("[E4-ABL] phase 0: AW-NSGA-II + fixed-weight NSGA-II-4obj "
              f"initial plans ({len(seeds)} seeds)...", flush=True)
    sc_cache: dict[int, object] = {}
    plans: dict = {"aw": {}, "nsga2": {}, "scenarios": sc_cache}
    aw_f1 = []
    for s in seeds:
        sc = make_scenario(s)
        sc_cache[s] = sc
        res_aw = run_algorithm("AW-NSGA-II", sc, s)
        plans["aw"][s] = res_aw.best_assign
        aw_f1.append(res_aw.best_F[0])
        res_fixed = nsga2_core(sc, s, obj_idx=(0, 1, 2, 3),
                               name="NSGA-II-4obj", weight_fn=None,
                               use_crowding=True, budget=EvalBudget())
        plans["nsga2"][s] = res_fixed.best_assign
        if verbose:
            print(f"    seed {s:3d} AW f1={res_aw.best_F[0]:7.1f} "
                  f"fixed-4obj f1={res_fixed.best_F[0]:7.1f}", flush=True)

    # cross-check AW initial plans against e1 (managed venv must reproduce)
    e1_path = os.path.join(os.path.dirname(out_dir), "e1", "e1_metrics.csv")
    if os.path.exists(e1_path):
        e1_f1 = {}
        with open(e1_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            for line in f:
                parts = line.strip().split(",")
                d = dict(zip(header, parts))
                if d["algo"] == "AW-NSGA-II":
                    e1_f1[int(d["seed"])] = float(d["f1"])
        mismatch = [(s, v, e1_f1.get(s)) for s, v in zip(seeds, aw_f1)
                    if abs(v - e1_f1.get(s, v)) > 1e-6]
        print("[E4-ABL] AW initial-plan check vs e1_metrics.csv: "
              + ("exact match" if not mismatch else
                 f"WARNING mismatch: {mismatch[:5]}"), flush=True)

    # ---- phase 1: ablation runs (new strategies only) ----------------------
    raw_new: list[dict] = []
    for kind in scenarios:
        insts = make_instances(kind)
        for k, inst in enumerate(insts):
            inst = dict(inst)
            inst["_idx"] = k
            inst["_nominal"] = (k == 0)
            for s in seeds:
                for strategy in NEW_STRATEGIES:
                    raw_new.append(run_ablation_scenario(kind, s, inst,
                                                         strategy, plans))
            if verbose:
                print(f"  [{kind}] instance {k}: done", flush=True)

    # ---- phase 2: merge with published rows and persist --------------------
    raw_pub = load_published(seeds, scenarios,
                             os.path.join(out_dir, "e4_raw.csv"))
    raw = raw_pub + raw_new
    _write_csv(os.path.join(out_dir, "e4_ablation_raw.csv"), raw)
    agg = aggregate(raw)
    _write_csv(os.path.join(out_dir, "e4_ablation_agg.csv"), agg)

    meta = {
        "strategies": ["EHCO", "noFLC", "noAW", "A*-GA"],
        "seeds": seeds, "scenarios": scenarios,
        "n_instances": 5, "n_repeats": len(seeds),
        "replan_weight_provider": {
            "EHCO": "FLC", "noFLC": "fixed (weight_fn=None, crowding)",
            "noAW": "FLC", "A*-GA": "no replanning"},
        "initial_plan": {
            "EHCO": "AW-NSGA-II", "noFLC": "AW-NSGA-II",
            "noAW": "NSGA-II-4obj fixed-weight", "A*-GA": "A*-GA"},
        "published_e4_rows_reused": True,
    }
    with open(os.path.join(out_dir, "e4_ablation_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=1, default=str)
    if verbose:
        print(f"[E4-ABL] total wall: {time.perf_counter()-t_start:.1f}s; "
              f"{len(raw_new)} new runs, {len(raw_pub)} reused",
              flush=True)
    return {"n_seeds": len(seeds), "n_new_runs": len(raw_new),
            "n_total_runs": len(raw), "out_dir": out_dir}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="comma list, e.g. 1,2,3")
    ap.add_argument("--scenarios", default=None,
                    help="comma list: congestion,fault,urgent")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    scenarios = args.scenarios.split(",") if args.scenarios else None
    run_e4_ablation(seeds=seeds, scenarios=scenarios, out_dir=args.out)


if __name__ == "__main__":
    main()
