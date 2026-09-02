"""C20: compound-disturbance Scenario D (expert pre-review, gap #2).

Motivation
----------
Every published E4 dynamic run injects exactly ONE event.  Real warehouses
experience superposed / cascading disturbances (congestion + equipment
fault + urgent orders within the same mission).  Scenario D injects a
strict cascade -- aisle congestion at t_c, an equipment fault (thermal or
vibration) at t_f > t_c, and three urgent orders at t_u > t_f -- and
compares four controllers on the SAME 30 seeds x 5 event instances:

  EHCO     : AW-NSGA-II initial plan + FLC-weighted closed-loop replanning
  A*-GA    : A*-GA initial plan, open loop (no replanning)
  noFLC    : AW-NSGA-II initial plan + fixed-weight NSGA-II replanning
             (the closed-loop NSGA-II rung of the attribution ladder)
  moead-cl : AW-NSGA-II initial plan + MOEA/D replanning (e_c19 controller)

Instance 0 is the nominal timeline (t_c = 150 s, aisle 6, 70% reduction;
thermal fault at t_f = 180 s; 3 urgent orders at t_u = 220 s); instances
1..4 randomise the onset times (strict cascade, 20-60 s gaps), aisle,
severity and fault type.  The replanning budgets, trigger logic, incumbent
guard and congestion-aware distance inflation are identical to E4.

Outputs (sim/results/e_c20/):
  e_c20_raw.csv    per (strategy, seed, instance) rows
  e_c20_agg.csv    mean +- std per (strategy, metric)
  e_c20_stats.csv  paired two-sided Wilcoxon (deg_pct, f1_event) vs EHCO
  e_c20_meta.json  protocol dump

Run:  python -m sim.experiments.e_c20 [--strategies EHCO,A*-GA,noFLC,moead-cl]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict

import numpy as np

from .. import constants as C
from ..algorithms import run_algorithm
from ..experiments.e4_dynamic import (DynamicSimulator, SEEDS, make_instances)
from ..experiments.e_c19 import MoeadSimulator
from ..metrics import cohen_dz, paired_wilcoxon
from ..scenario import make_scenario

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "e_c20")
STRATEGIES = ("EHCO", "A*-GA", "noFLC", "moead-cl")


def run_compound(seed: int, inst: dict, strategy: str, plans: dict) -> dict:
    """One (seed, instance, strategy) compound run."""
    sc = plans["scenarios"][seed]
    if strategy == "A*-GA":
        assign, closed = plans["astar"][seed], False
        sim = DynamicSimulator(sc, assign)
    elif strategy == "noFLC":
        assign, closed = plans["aw"][seed], True
        sim = DynamicSimulator(sc, assign, use_flc=False)
    elif strategy == "nocostadapt":
        # e_c22 ablation: replanning without the congestion-aware distance
        # inflation (the local replanner sees the static tables)
        assign, closed = plans["aw"][seed], True
        sim = DynamicSimulator(sc, assign)
        sim.congestion_aware_replan = False
    elif strategy == "moead-cl":
        assign, closed = plans["aw"][seed], True
        sim = MoeadSimulator(sc, assign)
    else:  # EHCO
        assign, closed = plans["aw"][seed], True
        sim = DynamicSimulator(sc, assign)

    ev = dict(inst)
    ev["rng"] = np.random.default_rng(seed * 1000 + 777)
    base = sim.run(event=None, closed_loop=False)
    out = sim.run(event=ev, closed_loop=closed)

    f1_base, f1_ev = base["makespan"], out["makespan"]
    deg = (f1_ev - f1_base) / f1_base * 100.0 if f1_base > 0 else float("nan")
    replans = [r for r in out["replan_info"] if not r.get("skipped", False)]
    row = {
        "scenario": "compound", "strategy": strategy,
        "seed": seed, "instance": inst.get("_idx", 0),
        "t_c": ev["t_c"], "t_f": ev["t_f"], "t_u": ev["t_u"],
        "fault_type": ev["fault_type"], "aisle": ev["aisle"],
        "reduction": ev["reduction"],
        "f1_base": f1_base, "f1_event": f1_ev, "deg_pct": deg,
        "timeout": int(out["timeout"]),
        "n_done": out["n_done"], "n_total": out["n_total"],
        "completion_pct": out["n_done"] / out["n_total"] * 100.0,
        "n_replans": out["n_replans"],
        "response_ms": replans[0]["wall_ms"] if replans else "",
        "response_ms_total": float(sum(r["wall_ms"] for r in replans))
        if replans else 0.0,
        "replan_evals": int(sum(r["evals"] for r in replans)) if replans else 0,
        "warn_t": out["warn_t"], "warn_agv": out["warn_agv"],
        "maint_enter": out["maint_enter"],
        "detected": int(out["warn_agv"] == out["fault_agv"]),
        "min_h": out["min_h"], "n_agv": C.N_AGV,
    }
    return row


def _write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)", flush=True)


def run_c20(seeds: list[int] | None = None,
            strategies: list[str] | None = None,
            verbose: bool = True) -> dict:
    seeds = seeds or SEEDS
    strategies = strategies or list(STRATEGIES)
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.perf_counter()

    # ---- phase 0: initial plans -------------------------------------------
    if verbose:
        print(f"[C20] phase 0: initial plans ({len(seeds)} seeds)...",
              flush=True)
    plans: dict = {"aw": {}, "astar": {}, "scenarios": {}}
    need_astar = "A*-GA" in strategies
    for s in seeds:
        sc = make_scenario(s)
        plans["scenarios"][s] = sc
        plans["aw"][s] = run_algorithm("AW-NSGA-II", sc, s).best_assign
        if need_astar:
            plans["astar"][s] = run_algorithm("A*-GA", sc, s).best_assign
        if verbose:
            print(f"    seed {s:3d} done", flush=True)

    # ---- phase 1: compound runs --------------------------------------------
    raw: list[dict] = []
    for k, inst in enumerate(make_instances("compound")):
        inst = dict(inst, _idx=k)
        for s in seeds:
            for strat in strategies:
                raw.append(run_compound(s, inst, strat, plans))
        if verbose:
            print(f"  [compound] instance {k}: done", flush=True)

    # merge with any previously computed rows for strategies not in this run
    raw_path = os.path.join(RESULTS, "e_c20_raw.csv")
    if os.path.exists(raw_path):
        have = {(r["strategy"], r["seed"], r["instance"]) for r in raw}
        with open(raw_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r["strategy"], int(r["seed"]), int(r["instance"])) \
                        not in have:
                    raw.append(r)
    _write_csv(raw_path, raw)

    # ---- phase 2: aggregation ------------------------------------------------
    def _mstats(vals):
        vals = np.asarray(vals, dtype=float)
        return (float(vals.mean()),
                float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)

    by: dict[str, list[dict]] = defaultdict(list)
    for r in raw:
        by[r["strategy"]].append(r)
    agg_rows = []
    METRICS = ("deg_pct", "f1_event", "n_replans", "response_ms_total",
               "completion_pct", "min_h", "warn_t")
    for strat, rows in sorted(by.items()):
        for metric in METRICS:
            vals = [float(r[metric]) for r in rows
                    if r.get(metric) not in ("", None)]
            if vals:
                m, s_ = _mstats(vals)
                agg_rows.append({"strategy": strat, "metric": metric,
                                 "mean": m, "std": s_, "n": len(vals)})
    _write_csv(os.path.join(RESULTS, "e_c20_agg.csv"), agg_rows)

    # ---- phase 3: paired stats vs EHCO ---------------------------------------
    bykey = {(r["strategy"], int(r["seed"]), int(r["instance"])): r
             for r in raw if isinstance(r.get("seed"), int)}
    stat_rows = []
    for strat in strategies:
        if strat == "EHCO":
            continue
        for metric in ("deg_pct", "f1_event", "n_replans", "min_h"):
            a, b = [], []
            for r in raw:
                if r["strategy"] != strat or not isinstance(r.get("seed"), int):
                    continue
                pa = bykey.get(("EHCO", r["seed"], r["instance"]))
                if pa is None:
                    continue
                a.append(float(pa[metric]))
                b.append(float(r[metric]))
            if len(a) < 5:
                continue
            p = paired_wilcoxon(np.array(b), np.array(a),
                                alternative="two-sided")
            stat_rows.append({"strategy_a": "EHCO", "strategy_b": strat,
                              "metric": metric, "mean_a": float(np.mean(a)),
                              "mean_b": float(np.mean(b)), "n": len(a),
                              "wilcoxon_p": p,
                              "cohen_dz": cohen_dz(np.array(b),
                                                   np.array(a))})
    _write_csv(os.path.join(RESULTS, "e_c20_stats.csv"), stat_rows)

    insts = make_instances("compound")
    meta = {
        "seeds": seeds, "strategies": strategies,
        "instances": insts,
        "protocol": ("strict cascade congestion (t_c) -> equipment fault "
                     "(t_f > t_c) -> 3 urgent orders (t_u > t_f); replanning "
                     "budgets / triggers / incumbent guard identical to E4"),
        "n_runs": len(raw),
    }
    with open(os.path.join(RESULTS, "e_c20_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=1, default=str)
    if verbose:
        print(f"[C20] total wall: {time.perf_counter()-t0:.1f}s; "
              f"{len(raw)} runs", flush=True)
    return {"n_seeds": len(seeds), "n_runs": len(raw), "out_dir": RESULTS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--strategies", default=None,
                    help="comma list: EHCO,A*-GA,noFLC,moead-cl")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    strategies = args.strategies.split(",") if args.strategies else None
    run_c20(seeds=seeds, strategies=strategies)


if __name__ == "__main__":
    main()
