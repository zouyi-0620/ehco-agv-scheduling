"""C18: randomised-severity near-threshold fault injection (reviewer C-group).

Motivation
----------
The published E4 fault protocol injects a deterministic worst-case fault
(load rate 0.4 -> 1.0 plus thermal runaway to a fixed 120 degC steady
state), which ALWAYS drives the EWMA health of the faulty AGV below
h_safe = 0.6: detection F1 = 1.00 is partly an artefact of the injection
pipeline (reviewer concern).  C18 stress-tests the detector:

1. Randomised severity: the fault's thermal steady state is drawn from a
   grid spanning the detection threshold -- {105, 109, 113, 117, 121, 125}
   degC.  (Analytically h_inst = 0.6 corresponds to ~113.7 degC steady
   state, so mild faults near 105-113 degC may never cross the warning
   line -> missed detections / recall < 1.)
2. No-fault high-stress distractor: in every run one healthy AGV (the
   SECOND most-loaded vehicle of the initial plan, so it is usually
   distinct from the fault target) starts with h0 = 0.62, just above the
   warning line, and carries its full published schedule -- a near-
   threshold vehicle under sustained load.  Warnings on such a vehicle are
   false positives (specificity stress).

Protocol per run: 30 published seeds x 6 severity levels = 180 runs,
thermal faults only, t_inj ~ U[60, 200] s (E4 adaptive window), fault
target = the AGV in a loaded run with the largest remaining workload at
t_inj (E4 adaptive rule), initial plan = published AW-NSGA-II, closed-loop
EHCO controller (replanning only if the warning fires).

Detection semantics follow E4: `detected` = the FIRST warning of the run
is on the faulty AGV; `false_pos` = number of non-faulty AGVs that warn
(distractor included when it is not the fault target).  `faulty_ever`
additionally records whether the faulty AGV warned at any time.

Outputs (sim/results/e_c18/):
  e_c18_raw.csv     per (seed, severity): detection + performance row
  e_c18_agg.csv     detection rate / F1 / latency per severity level
  e_c18_meta.json   protocol dump

Run from the project root:  python -m sim.experiments.e_c18
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np

from .. import constants as C
from ..algorithms import run_algorithm
from ..experiments.e4_dynamic import DynamicSimulator
from ..scenario import AGVInit, Scenario, make_scenario

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "e_c18")
SEEDS = list(range(1, 31))
SEVERITIES = [105.0, 109.0, 111.0, 113.0, 117.0, 121.0, 125.0]
DISTRACTOR_H0 = 0.62          # just above h_safe, below the published U[0.7,1]


def make_c18_scenario(sc: Scenario, distractor: int) -> Scenario:
    """Shallow copy with the distractor AGV's h0 set to the stress value."""
    agvs = [AGVInit(a.agv_id, a.start_node, a.h0) for a in sc.agvs]
    agvs[distractor] = AGVInit(agvs[distractor].agv_id,
                               agvs[distractor].start_node, DISTRACTOR_H0)
    return Scenario(seed=sc.seed, warehouse=sc.warehouse,
                    tasks=sc.tasks, agvs=agvs)


def run_one(seed: int, severity: float, plans: dict) -> dict:
    sc0 = plans["scenarios"][seed]
    assign = plans["aw"][seed]
    # distractor: second most-loaded AGV of the initial plan
    counts = np.bincount(assign, minlength=len(sc0.agvs))
    order = np.argsort(-counts)
    distractor = int(order[1]) if len(order) > 1 else int(order[0])

    sc = make_c18_scenario(sc0, distractor)
    ev_rng = np.random.default_rng(seed * 1000 + 777)
    t_inj = float(ev_rng.uniform(60.0, 200.0))
    ev = {"kind": "fault", "t_inj": t_inj, "agv": None,
          "fault_type": "thermal", "rng": ev_rng}

    sim = DynamicSimulator(sc, assign)          # closed-loop EHCO path
    sim.fault_t_ss = severity                    # randomised severity
    # no-event baseline: pure distractor false-alarm test (no fault present)
    base = sim.run(event=None, closed_loop=False)
    base_distractor_warned = int(distractor in base["warned_indices"])
    base_n_warned = int(base["n_warned"])
    out = sim.run(event=ev, closed_loop=True)

    f1_base, f1_ev = base["makespan"], out["makespan"]
    deg = (f1_ev - f1_base) / f1_base * 100.0 if f1_base > 0 else float("nan")
    distractor_hit = int(out["fault_agv"] == distractor)
    detected = int(out["warn_agv"] == out["fault_agv"]) \
        if out["warn_agv"] is not None else 0
    faulty_ever = int(out["fault_agv"] is not None
                      and out["fault_agv"] in out["warned_indices"])
    false_pos = int(sum(1 for i in out["warned_indices"]
                        if i != out["fault_agv"]))
    distractor_warned = int(not distractor_hit
                            and distractor in out["warned_indices"])
    latency = (out["warn_t"] - t_inj) if (out["warn_t"] is not None
                                          and out["fault_agv"] is not None
                                          and out["warn_agv"] == out["fault_agv"]) \
        else ""
    return {
        "seed": seed, "severity": severity, "t_inj": t_inj,
        "distractor": distractor, "distractor_hit": distractor_hit,
        "fault_agv": out["fault_agv"], "fault_type": "thermal",
        "detected": detected, "faulty_ever": faulty_ever,
        "false_pos": false_pos, "distractor_warned": distractor_warned,
        "warn_t": out["warn_t"], "latency_s": latency,
        "base_distractor_warned": base_distractor_warned,
        "base_n_warned": base_n_warned,
        "f1_base": f1_base, "f1_event": f1_ev, "deg_pct": deg,
        "timeout": int(out["timeout"]),
        "maint_enter": out["maint_enter"],
        "n_warned": out["n_warned"],
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


def run_c18(seeds: list[int] | None = None, verbose: bool = True) -> dict:
    seeds = seeds or SEEDS
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.perf_counter()

    plans = {"aw": {}, "scenarios": {}}
    for s in seeds:
        sc = make_scenario(s)
        plans["scenarios"][s] = sc
        plans["aw"][s] = run_algorithm("AW-NSGA-II", sc, s).best_assign

    raw = []
    for sev in SEVERITIES:
        for s in seeds:
            raw.append(run_one(s, sev, plans))
        if verbose:
            det = np.mean([r["detected"] for r in raw
                           if r["severity"] == sev])
            print(f"  severity {sev:.0f} degC: first-warning detection "
                  f"{det*100:.0f}% ({len(seeds)} seeds)", flush=True)

    _write_csv(os.path.join(RESULTS, "e_c18_raw.csv"), raw)

    # ---- aggregation per severity -----------------------------------------
    agg_rows = []
    for sev in SEVERITIES:
        rows = [r for r in raw if r["severity"] == sev]
        det = np.array([r["detected"] for r in rows])
        ever = np.array([r["faulty_ever"] for r in rows])
        fp = np.array([r["false_pos"] for r in rows])
        dw = np.array([r["distractor_warned"] for r in rows
                       if not r["distractor_hit"]])
        lat = np.array([float(r["latency_s"]) for r in rows
                        if r["latency_s"] != ""], dtype=float)
        # precision / recall / F1 on first-warning semantics
        tp = int(det.sum()); fn = int((1 - det).sum())
        fpp = int(fp.sum())
        prec = tp / (tp + fpp) if (tp + fpp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
        deg = np.array([float(r["deg_pct"]) for r in rows], dtype=float)
        agg_rows.append({
            "severity": sev, "n": len(rows),
            "det_rate_first": float(det.mean()),
            "det_rate_ever": float(ever.mean()),
            "false_alarm_per_run": float(fp.mean()),
            "precision_first": prec, "recall_first": rec, "f1_first": f1,
            "distractor_warn_rate": float(dw.mean()) if len(dw) else "",
            "distractor_n": len(dw),
            "latency_mean": float(lat.mean()) if len(lat) else "",
            "latency_std": float(lat.std(ddof=1)) if len(lat) > 1 else "",
            "deg_pct_mean": float(deg.mean()),
            "deg_pct_std": float(deg.std(ddof=1)),
        })
    _write_csv(os.path.join(RESULTS, "e_c18_agg.csv"), agg_rows)

    meta = {
        "seeds": seeds, "severities": SEVERITIES,
        "distractor_h0": DISTRACTOR_H0,
        "distractor_rule": "second most-loaded AGV of the initial AW plan",
        "fault": "thermal, load rate 0.4->1.0, steady state = severity",
        "t_inj": "U[60,200] s, event rng default_rng(seed*1000+777)",
        "target_rule": "E4 adaptive (largest remaining workload in loaded run)",
        "detection_semantics": "detected = first warning on the faulty AGV "
                               "(E4 convention); faulty_ever = any-time",
    }
    with open(os.path.join(RESULTS, "e_c18_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    if verbose:
        print(f"[C18] total wall: {time.perf_counter()-t0:.1f}s; "
              f"{len(raw)} runs", flush=True)
    return {"n_seeds": len(seeds), "n_runs": len(raw), "out_dir": RESULTS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="comma list, e.g. 1,2,3")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    run_c18(seeds=seeds)


if __name__ == "__main__":
    main()
