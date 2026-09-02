"""C21: dynamic beta sensitivity (expert pre-review, gap #3).

Motivation
----------
The published E6 sensitivity analysis perturbed EWMA_BETA only in the
STATIC scenario, where f4 is inactive and the optimisation solution cannot
change -- so it never tested whether EHCO's *dynamic decisions* are robust
to beta.  The expert pre-review asks for the dynamic counterpart: in the
fault scenario beta controls how fast the cumulative health h_cum tracks
the physical degradation, hence the warning time, the maintenance-entry
time and the resulting makespan degradation.

Protocol
--------
Fault scenario (the only scenario whose decisions depend on beta through
the h_cum < 0.6 trigger), EHCO controller (AW-NSGA-II initial plan +
FLC-weighted replanning, published configuration), 5 event instances x 30
seeds per beta value:

  beta = 5e-4   (half-life ~69 s, slow tracker)   -- new
  beta = 1e-3   (half-life ~35 s, published)      -- reused from e4_raw.csv
  beta = 2e-3   (half-life ~17 s)                 -- new
  beta = 5e-3   (half-life ~7 s, fast tracker)    -- new

C.EWMA_BETA is patched around each run (it is read at call time inside
DynamicSimulator._health_step and objectives.evaluate_population, so the
patch is effective for both the monitoring layer and the replanning
objective).  The initial plans are computed once: static f4 is identically
zero for every beta (verified at start-up on one seed), so the plans are
beta-invariant.

Outputs (sim/results/e_c21/):
  e_c21_raw.csv     new rows (beta column + lead_time + min_h)
  e_c21_agg.csv     mean +- std per (beta, metric)
  e_c21_stats.csv   paired two-sided Wilcoxon vs the published beta=1e-3 rows
  e_c21_meta.json   protocol dump

Run:  python -m sim.experiments.e_c21 [--betas 0.0005,0.002,0.005]
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
from ..metrics import cohen_dz, paired_wilcoxon
from ..scenario import make_scenario

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "e_c21")
PUBLISHED_E4 = os.path.join(os.path.dirname(__file__), "..", "..",
                            "results", "e4", "e4_raw.csv")
NEW_BETAS = (5e-4, 2e-3, 5e-3)


def run_beta(seed: int, inst: dict, plans: dict, beta: float) -> dict:
    """One (seed, instance) fault run under a patched EWMA_BETA."""
    sc = plans["scenarios"][seed]
    assign = plans["aw"][seed]
    ev = dict(inst)
    ev["rng"] = np.random.default_rng(seed * 1000 + 777)

    old = C.EWMA_BETA
    C.EWMA_BETA = beta
    try:
        sim = DynamicSimulator(sc, assign)
        base = sim.run(event=None, closed_loop=False)
        out = sim.run(event=ev, closed_loop=True)
    finally:
        C.EWMA_BETA = old

    f1_base, f1_ev = base["makespan"], out["makespan"]
    deg = (f1_ev - f1_base) / f1_base * 100.0 if f1_base > 0 else float("nan")
    lead = (out["warn_t"] - ev["t_inj"]) if out["warn_t"] is not None else None
    row = {
        "scenario": "fault", "strategy": "EHCO", "beta": beta,
        "seed": seed, "instance": inst.get("_idx", 0),
        "t_inj": ev["t_inj"], "fault_type": ev["fault_type"],
        "f1_base": f1_base, "f1_event": f1_ev, "deg_pct": deg,
        "timeout": int(out["timeout"]),
        "response_ms": out["replan_walls_ms"][0] if out["replan_walls_ms"] else "",
        "warn_t": out["warn_t"], "lead_time": lead,
        "warn_agv": out["warn_agv"],
        "detected": int(out["warn_agv"] == out["fault_agv"]),
        "false_pos": int(sum(1 for i in out["warned_indices"]
                             if i != out["fault_agv"])),
        "maint_enter": out["maint_enter"], "min_h": out["min_h"],
        "replan_applied": int(bool(out["replan_info"])
                              and not out["replan_info"][0].get("skipped", False)
                              and out["replan_info"][0].get("applied", True)),
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


def run_c21(seeds: list[int] | None = None,
            betas: list[float] | None = None,
            verbose: bool = True) -> dict:
    seeds = seeds or SEEDS
    betas = betas or list(NEW_BETAS)
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.perf_counter()

    # ---- phase 0: initial plans + beta-invariance check ---------------------
    if verbose:
        print(f"[C21] phase 0: initial plans ({len(seeds)} seeds)...",
              flush=True)
    plans: dict = {"aw": {}, "scenarios": {}}
    for s in seeds:
        sc = make_scenario(s)
        plans["scenarios"][s] = sc
        plans["aw"][s] = run_algorithm("AW-NSGA-II", sc, s).best_assign
        if verbose:
            print(f"    seed {s:3d} done", flush=True)

    # verify on one seed that the AW initial plan is beta-invariant
    # (static f4 == 0 for every beta -> identical selection)
    s0 = seeds[0]
    sc0 = plans["scenarios"][s0]
    f1_ref = run_algorithm("AW-NSGA-II", sc0, s0).best_F[0]
    for beta in betas:
        old = C.EWMA_BETA
        C.EWMA_BETA = beta
        try:
            f1_b = run_algorithm("AW-NSGA-II", sc0, s0).best_F[0]
        finally:
            C.EWMA_BETA = old
        ok = abs(f1_b - f1_ref) < 1e-9
        print(f"[C21] beta-invariance of initial plan (seed {s0}, "
              f"beta={beta:g}): {'exact' if ok else 'DIFFERS'}", flush=True)
        if not ok:
            raise RuntimeError("initial plan is beta-dependent; plans must "
                               "be recomputed per beta (unexpected: static "
                               "f4 is identically zero)")

    # ---- phase 1: beta runs ---------------------------------------------------
    raw: list[dict] = []
    for beta in betas:
        for k, inst in enumerate(make_instances("fault")):
            inst = dict(inst, _idx=k)
            for s in seeds:
                raw.append(run_beta(s, inst, plans, beta))
            if verbose:
                print(f"  [beta={beta:g}] instance {k}: done", flush=True)
    _write_csv(os.path.join(RESULTS, "e_c21_raw.csv"), raw)

    # ---- phase 2: aggregation (new betas + published beta=1e-3) ---------------
    pub: dict[tuple, dict] = {}
    with open(PUBLISHED_E4, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["scenario"] == "fault" and r["strategy"] == "EHCO":
                pub[(int(r["seed"]), int(r["instance"]))] = r

    def _mstats(vals):
        vals = np.asarray(vals, dtype=float)
        return (float(vals.mean()),
                float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)

    allrows: dict[float, list[dict]] = defaultdict(list)
    for r in raw:
        allrows[round(float(r["beta"]), 6)].append(r)
    pub_rows = []
    for (s, i), r in pub.items():
        pub_rows.append({"beta": 1e-3, "seed": s, "instance": i,
                         "deg_pct": float(r["deg_pct"]),
                         "f1_event": float(r["f1_event"]),
                         "warn_t": float(r["warn_t"])
                         if r["warn_t"] not in ("", "None") else None,
                         "detected": int(r.get("detected", 1))
                         if r.get("detected") != "" else None,
                         "maint_enter": float(r["maint_enter"])
                         if r.get("maint_enter") not in ("", "None") else None,
                         "t_inj": float(r["t_ev"])})
    allrows[0.001] = pub_rows

    agg_rows = []
    for beta, rows in sorted(allrows.items()):
        for metric in ("deg_pct", "f1_event", "warn_t", "detected",
                       "maint_enter"):
            vals = [float(r[metric]) for r in rows
                    if r.get(metric) not in ("", None)]
            if vals:
                m, s_ = _mstats(vals)
                agg_rows.append({"beta": beta, "metric": metric,
                                 "mean": m, "std": s_, "n": len(vals)})
    _write_csv(os.path.join(RESULTS, "e_c21_agg.csv"), agg_rows)

    # ---- phase 3: paired stats vs published beta=1e-3 --------------------------
    stat_rows = []
    for beta in betas:
        beta_rows = {(r["seed"], r["instance"]): r for r in raw
                     if abs(float(r["beta"]) - beta) < 1e-12}
        for metric in ("deg_pct", "f1_event", "warn_t"):
            a, b = [], []
            for (s, i), r in beta_rows.items():
                pa = pub.get((s, i))
                if pa is None or r.get(metric) in ("", None) \
                        or pa.get(metric) in ("", None, "None"):
                    continue
                a.append(float(pa[metric]))
                b.append(float(r[metric]))
            if len(a) < 5:
                continue
            p = paired_wilcoxon(np.array(b), np.array(a),
                                alternative="two-sided")
            stat_rows.append({"beta": beta, "metric": metric,
                              "mean_beta001": float(np.mean(a)),
                              "mean_new": float(np.mean(b)),
                              "n": len(a), "wilcoxon_p": p,
                              "cohen_dz": cohen_dz(np.array(b),
                                                   np.array(a))})
    _write_csv(os.path.join(RESULTS, "e_c21_stats.csv"), stat_rows)

    meta = {
        "seeds": seeds, "betas_new": betas, "beta_published": 1e-3,
        "scenario": "fault", "controller": "EHCO (published configuration)",
        "half_lives_s": {f"{b:g}": round(float(np.log(0.5)
                                     / np.log(1 - b)) * C.DT, 1)
                         for b in list(betas) + [1e-3]},
        "patch": "sim.constants.EWMA_BETA patched around each run "
                 "(read at call time in _health_step and evaluate_population)",
        "published_e4": os.path.relpath(PUBLISHED_E4),
    }
    with open(os.path.join(RESULTS, "e_c21_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    if verbose:
        print(f"[C21] total wall: {time.perf_counter()-t0:.1f}s; "
              f"{len(raw)} new runs", flush=True)
    return {"n_seeds": len(seeds), "n_runs": len(raw), "out_dir": RESULTS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--betas", default=None,
                    help="comma list, e.g. 0.0005,0.002,0.005")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    betas = [float(x) for x in args.betas.split(",")] if args.betas else None
    run_c21(seeds=seeds, betas=betas)


if __name__ == "__main__":
    main()
