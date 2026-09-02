"""C22: dynamic path-cost ablation (expert pre-review, gap #4).

Motivation
----------
The static ablation (-impAstar) was bit-identical to Full EHCO, and the
manmanuscript explains this as "the multi-cost fusion evaluation affects
path selection only when the path layer must respond to changed
conditions".  The expert pre-review asks for the dynamic counterpart:
Standard A* vs Improved A* under dynamic events.  Two facts matter here:

1. STRUCTURAL DEGENERACY OF THE STATIC FUSION FORM.  Under the calibrated
   parameterization the fused cell cost (1 + w_e * e_avg) is uniform over
   the grid and the health term w_h * g_health is zero for h >= h_safe,
   so the min-fused-cost paths are geometrically identical to the standard
   shortest paths: dist_improved == dist_standard element-wise (verified
   numerically at start-up, 9409/9409 entries equal).  Any cost that is
   proportional to path length leaves the argmin unchanged, so swapping
   the tables cannot alter plans by construction.

2. THE DYNAMIC COST LAYER IS THE CONGESTION-AWARE INFLATION.  The only
   place where the path-cost layer responds to changed conditions is the
   replanning distance matrix: node pairs whose shortest path crosses the
   congested aisle band cost D / (1 - reduction) inside the local
   replanner (plus the incumbent adoption guard evaluated under the same
   model).  Removing it yields a replanner that re-sequences blindly.

Variants
--------
  stdA*       : standard-A* tables in BOTH the planner and the executor
                (EvalConfig.use_improved_astar=False and a scenario copy
                whose dist_improved is replaced by dist_standard).
                Expected and verified: bit-identical trajectories to the
                published EHCO rows (structural degeneracy, fact 1).
  nocostadapt : congestion_aware_replan=False -- the local replanner keeps
                the static tables and the unconditional adoption path.
                Run on the congestion scenario and on the compound
                Scenario D; measures the value of event-responsive path
                costs (fact 2).

Outputs (sim/results/e_c22/):
  e_c22_raw.csv          all new rows
  e_c22_identity.csv     stdA* vs published EHCO bit-identity check
  e_c22_stats.csv        paired two-sided Wilcoxon (nocostadapt vs EHCO)
  e_c22_meta.json        protocol dump

Run:  python -m sim.experiments.e_c22 [--variants stdA*,nocostadapt]
"""
from __future__ import annotations

import argparse
import copy as _copy
import csv
import json
import os
import time
from collections import defaultdict

import numpy as np

from .. import constants as C
from ..algorithms import run_algorithm
from ..experiments.e4_dynamic import (DynamicSimulator, SEEDS, make_instances)
from ..experiments.e_c20 import run_compound
from ..metrics import cohen_dz, paired_wilcoxon
from ..objectives import EvalConfig
from ..scenario import Scenario, make_scenario

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "e_c22")
PUBLISHED_E4 = os.path.join(os.path.dirname(__file__), "..", "..",
                            "results", "e4", "e4_raw.csv")
C20_RAW = os.path.join(os.path.dirname(__file__), "..", "results", "e_c20",
                       "e_c20_raw.csv")
VARIANTS = ("stdA*", "nocostadapt")


def _std_scenario(sc) -> Scenario:
    """Scenario copy whose improved table IS the standard table."""
    wh2 = _copy.copy(sc.warehouse)
    wh2.dist_improved = wh2.dist_standard
    return Scenario(seed=sc.seed, warehouse=wh2, tasks=sc.tasks,
                    agvs=sc.agvs)


def run_stdastar(kind: str, seed: int, inst: dict, plans: dict) -> dict:
    """One stdA* run: standard tables in planner AND executor."""
    sc = plans["scenarios"][seed]
    sc2 = _std_scenario(sc)
    sim = DynamicSimulator(sc2, plans["aw"][seed],
                           cfg=EvalConfig(use_improved_astar=False))
    ev = dict(inst)
    ev["rng"] = np.random.default_rng(seed * 1000 + 777)
    base = sim.run(event=None, closed_loop=False)
    out = sim.run(event=ev, closed_loop=True)
    f1_base, f1_ev = base["makespan"], out["makespan"]
    deg = (f1_ev - f1_base) / f1_base * 100.0 if f1_base > 0 else float("nan")
    return {
        "scenario": kind, "strategy": "stdA*", "seed": seed,
        "instance": inst.get("_idx", 0),
        "f1_base": f1_base, "f1_event": f1_ev, "deg_pct": deg,
        "warn_t": out["warn_t"], "min_h": out["min_h"],
        "n_replans": out["n_replans"],
    }


def run_nocostadapt(kind: str, seed: int, inst: dict, plans: dict) -> dict:
    """One nocostadapt run (congestion or compound kind)."""
    if kind == "compound":
        return run_compound(seed, inst, "nocostadapt", plans)
    sc = plans["scenarios"][seed]
    sim = DynamicSimulator(sc, plans["aw"][seed])
    sim.congestion_aware_replan = False
    ev = dict(inst)
    ev["rng"] = np.random.default_rng(seed * 1000 + 777)
    base = sim.run(event=None, closed_loop=False)
    out = sim.run(event=ev, closed_loop=True)
    f1_base, f1_ev = base["makespan"], out["makespan"]
    deg = (f1_ev - f1_base) / f1_base * 100.0 if f1_base > 0 else float("nan")
    return {
        "scenario": kind, "strategy": "nocostadapt", "seed": seed,
        "instance": inst.get("_idx", 0),
        "f1_base": f1_base, "f1_event": f1_ev, "deg_pct": deg,
        "warn_t": out["warn_t"], "min_h": out["min_h"],
        "n_replans": out["n_replans"],
        "response_ms": out["replan_walls_ms"][0]
        if out["replan_walls_ms"] else "",
        "replan_applied": int(bool(out["replan_info"])
                              and not out["replan_info"][0].get("skipped", False)
                              and out["replan_info"][0].get("applied", True)),
    }


def _write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)", flush=True)


def run_c22(seeds: list[int] | None = None,
            variants: list[str] | None = None,
            verbose: bool = True) -> dict:
    seeds = seeds or SEEDS
    variants = variants or list(VARIANTS)
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.perf_counter()

    # ---- phase -1: structural degeneracy check -------------------------------
    from ..warehouse import Warehouse
    wh = Warehouse.create()
    eq = bool(np.array_equal(wh.dist_standard, wh.dist_improved))
    print(f"[C22] dist_improved == dist_standard (element-wise): {eq} "
          f"({wh.dist_standard.size} entries)", flush=True)

    # ---- phase 0: initial plans -------------------------------------------
    if verbose:
        print(f"[C22] phase 0: AW-NSGA-II initial plans "
              f"({len(seeds)} seeds)...", flush=True)
    plans: dict = {"aw": {}, "scenarios": {}}
    for s in seeds:
        sc = make_scenario(s)
        plans["scenarios"][s] = sc
        plans["aw"][s] = run_algorithm("AW-NSGA-II", sc, s).best_assign
        if verbose:
            print(f"    seed {s:3d} done", flush=True)

    pub: dict[tuple, dict] = {}
    with open(PUBLISHED_E4, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["strategy"] == "EHCO":
                pub[(r["scenario"], int(r["seed"]), int(r["instance"]))] = r

    raw: list[dict] = []
    ident_rows: list[dict] = []

    # ---- phase 1a: stdA* bit-identity documentation ---------------------------
    if "stdA*" in variants:
        for kind in ("congestion", "fault"):
            for k, inst in enumerate(make_instances(kind)):
                inst = dict(inst, _idx=k)
                for s in seeds:
                    row = run_stdastar(kind, s, inst, plans)
                    raw.append(row)
                    pa = pub.get((kind, s, k))
                    if pa is not None:
                        ident_rows.append({
                            "scenario": kind, "seed": s, "instance": k,
                            "f1_event_stdA": row["f1_event"],
                            "f1_event_EHCO": float(pa["f1_event"]),
                            "abs_diff": abs(row["f1_event"]
                                            - float(pa["f1_event"])),
                            "deg_diff": abs(row["deg_pct"]
                                            - float(pa["deg_pct"])),
                        })
            if verbose:
                print(f"  [stdA*] {kind}: done", flush=True)
        dmax = max((r["abs_diff"] for r in ident_rows), default=float("nan"))
        print(f"[C22] stdA* vs published EHCO: max |f1_event diff| = {dmax}",
              flush=True)

    # ---- phase 1b: nocostadapt -------------------------------------------------
    if "nocostadapt" in variants:
        for k, inst in enumerate(make_instances("congestion")):
            inst = dict(inst, _idx=k)
            for s in seeds:
                raw.append(run_nocostadapt("congestion", s, inst, plans))
            if verbose:
                print(f"  [nocostadapt/congestion] instance {k}: done",
                      flush=True)
        if os.path.exists(C20_RAW):
            for k, inst in enumerate(make_instances("compound")):
                inst = dict(inst, _idx=k)
                for s in seeds:
                    raw.append(run_nocostadapt("compound", s, inst, plans))
                if verbose:
                    print(f"  [nocostadapt/compound] instance {k}: done",
                          flush=True)
        else:
            print("[C22] compound comparison skipped (run e_c20 first)",
                  flush=True)

    _write_csv(os.path.join(RESULTS, "e_c22_raw.csv"), raw)
    if ident_rows:
        _write_csv(os.path.join(RESULTS, "e_c22_identity.csv"), ident_rows)

    # ---- phase 2: paired stats (nocostadapt vs EHCO) ----------------------------
    c20_ehco: dict[tuple, dict] = {}
    if os.path.exists(C20_RAW):
        with open(C20_RAW, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["strategy"] == "EHCO":
                    c20_ehco[(int(r["seed"]), int(r["instance"]))] = r

    stat_rows = []
    for kind in ("congestion", "compound"):
        rows_n = [r for r in raw if r["strategy"] == "nocostadapt"
                  and r["scenario"] == kind]
        if not rows_n:
            continue
        for metric in ("deg_pct", "f1_event"):
            a, b = [], []
            for r in rows_n:
                if kind == "compound":
                    pa = c20_ehco.get((r["seed"], r["instance"]))
                else:
                    pa = pub.get((kind, r["seed"], r["instance"]))
                if pa is None:
                    continue
                a.append(float(pa[metric]))
                b.append(float(r[metric]))
            if len(a) < 5:
                continue
            p = paired_wilcoxon(np.array(b), np.array(a),
                                alternative="two-sided")
            stat_rows.append({"scenario": kind, "strategy_a": "EHCO",
                              "strategy_b": "nocostadapt", "metric": metric,
                              "mean_a": float(np.mean(a)),
                              "mean_b": float(np.mean(b)), "n": len(a),
                              "wilcoxon_p": p,
                              "cohen_dz": cohen_dz(np.array(b),
                                                   np.array(a))})
    _write_csv(os.path.join(RESULTS, "e_c22_stats.csv"), stat_rows)

    meta = {
        "seeds": seeds, "variants": variants,
        "dist_improved_equals_dist_standard": eq,
        "stdA_protocol": "standard tables in planner + executor "
                         "(EvalConfig.use_improved_astar=False)",
        "nocostadapt_protocol": "congestion_aware_replan=False: local "
                                "replanner keeps static tables, "
                                "unconditional adoption",
        "published_e4": os.path.relpath(PUBLISHED_E4),
        "c20_raw": os.path.relpath(C20_RAW),
    }
    with open(os.path.join(RESULTS, "e_c22_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    if verbose:
        print(f"[C22] total wall: {time.perf_counter()-t0:.1f}s; "
              f"{len(raw)} new runs", flush=True)
    return {"n_seeds": len(seeds), "n_runs": len(raw), "out_dir": RESULTS,
            "bit_identity_max_diff": max(
                (r["abs_diff"] for r in ident_rows), default=None)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--variants", default=None,
                    help="comma list: stdA*,nocostadapt")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    variants = args.variants.split(",") if args.variants else None
    run_c22(seeds=seeds, variants=variants)


if __name__ == "__main__":
    main()
