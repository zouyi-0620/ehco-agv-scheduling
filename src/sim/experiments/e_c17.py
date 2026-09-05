"""C17: health-agnostic replanning baseline (reviewer C-group, R1-M3/R2-M6).

Motivation
----------
In the published E4 dynamic experiments the only baseline is the open-loop
A*-GA plan (no replanning at all), so the closed-loop gain can only be
attributed to "replanning > not replanning" -- never to the health-awareness
of the EHCO controller.  C17 adds a replanning-capable, health-agnostic
closed-loop controller with IDENTICAL triggers and IDENTICAL optimisation
budgets as EHCO:

  agnostic      : at every EHCO replanning trigger (congestion onset, urgent
                  order insertion, platform health warning) re-optimise the
                  remaining tasks with the same NSGA-II core and the same
                  (50 x 80 local / 100 x 200 global) budgets, but optimising
                  (f1, f2, f3) ONLY -- no f4, no maintenance cost, no FLC
                  state feedback (crowding-based NSGA-II, the standard
                  health-blind formulation).  In the fault scenario the
                  controller does NOT exclude the faulty AGV: it reacts to
                  the same alert but has no health-aware response.
  agnostic-excl : fault scenario only -- same health-blind objective, but the
                  faulty AGV is excluded and sent to maintenance exactly as
                  EHCO does.  This isolates the health-aware OBJECTIVE from
                  the fault-response ACTION.

Decomposition of the closed-loop gain (fault scenario):
  EHCO           vs open-loop     : total closed-loop gain (published E4)
  agnostic-excl  vs open-loop     : gain from replanning + fault removal
  EHCO           vs agnostic-excl : gain from the health-aware objective/FLC
  agnostic       vs agnostic-excl : gain from excluding the faulty AGV
  agnostic       vs EHCO          : full health-awareness package effect

The initial plan is the SAME published AW-NSGA-II plan for every strategy
(the closed-loop policy is the only difference).  Event instances, event RNG
streams and the no-event baseline replay are identical to E4, so the
published EHCO and A*-GA rows in results/e4/e4_raw.csv are directly paired
with the new runs.

Outputs (sim/results/e_c17/):
  e_c17_raw.csv     new rows (same schema as e4_raw.csv)
  e_c17_agg.csv     mean +- std deg_pct / response / timeout per strategy
  e_c17_stats.csv   paired two-sided Wilcoxon (deg_pct) per scenario
  e_c17_meta.json   protocol dump

Run from the project root:  python -m sim.experiments.e_c17
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
from ..experiments.e4_dynamic import (DynamicSimulator, make_instances)
from ..metrics import cohen_dz, paired_wilcoxon
from ..objectives import EvalConfig
from ..scenario import make_scenario

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "e_c17")
_CODE_E4 = os.path.join(os.path.dirname(__file__), "..", "results", "e4", "e4_raw.csv")
_REL_E4 = os.path.join(os.path.dirname(__file__), "..", "..", "results", "e4", "e4_raw.csv")
PUBLISHED_E4 = _CODE_E4 if os.path.exists(_CODE_E4) else _REL_E4
SEEDS = list(range(1, 31))


# --------------------------------------------------------------------------
# health-agnostic controller
# --------------------------------------------------------------------------

class AgnosticSimulator(DynamicSimulator):
    """Closed-loop controller with replanning but without health awareness.

    Same triggers, same budgets, same incumbent guard and congestion-aware
    distance inflation as the published EHCO path; the optimiser is a
    standard 3-objective NSGA-II (crowding) over (f1, f2, f3)."""

    def __init__(self, sc, assign, exclude_faulty: bool = False):
        super().__init__(sc, assign,
                         cfg=EvalConfig(use_health_objective=False,
                                        use_maintenance_cost=False),
                         use_flc=False)
        self.replan_obj_idx = (0, 1, 2)
        self.exclude_faulty = exclude_faulty

    def _global_replan(self, rng, exclude_agv):
        if self.exclude_faulty:
            # identical fault response to EHCO (exclude + maintenance travel)
            return super()._global_replan(rng, exclude_agv)
        # health-blind response: full-fleet re-optimisation, the faulty AGV
        # stays in service (its protective derating still applies physically)
        sc_p = self._replan_scenario(exclude_agv=None)
        return self._replan(sc_p, rng, C.NP_GLOBAL, C.NP_GLOBAL * C.GMAX_GLOBAL,
                            "agnostic-global")


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def run_agnostic(kind: str, seed: int, inst: dict, plans: dict,
                 variant: str) -> dict:
    """One (seed, instance, agnostic-variant) run; schema mirrors e4_raw.csv."""
    sc = plans["scenarios"][seed]
    assign = plans["aw"][seed]                 # same initial plan as EHCO
    ev = dict(inst)
    ev["rng"] = np.random.default_rng(seed * 1000 + 777)

    sim = AgnosticSimulator(sc, assign,
                            exclude_faulty=(variant == "agnostic-excl"))
    base = sim.run(event=None, closed_loop=False)
    out = sim.run(event=ev, closed_loop=True)

    f1_base, f1_ev = base["makespan"], out["makespan"]
    deg = (f1_ev - f1_base) / f1_base * 100.0 if f1_base > 0 else float("nan")
    row = {
        "scenario": kind, "strategy": variant,
        "seed": seed, "instance": inst.get("_idx", 0),
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
        row["detected"] = int(out["warn_agv"] == out["fault_agv"])
        row["false_pos"] = int(sum(1 for i in out["warned_indices"]
                                   if i != out["fault_agv"]))
    return row


def _load_published(path: str) -> dict[tuple, dict]:
    pub = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pub[(r["scenario"], r["strategy"], int(r["seed"]),
                 int(r["instance"]))] = r
    return pub


def _write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)", flush=True)


def run_c17(seeds: list[int] | None = None, verbose: bool = True) -> dict:
    seeds = seeds or SEEDS
    os.makedirs(RESULTS, exist_ok=True)
    pub = _load_published(PUBLISHED_E4)
    t0 = time.perf_counter()

    # ---- phase 0: initial plans (identical to E4 / e1) --------------------
    if verbose:
        print("[C17] phase 0: AW-NSGA-II initial plans "
              f"({len(seeds)} seeds)...", flush=True)
    plans = {"aw": {}, "scenarios": {}}
    for s in seeds:
        sc = make_scenario(s)
        plans["scenarios"][s] = sc
        plans["aw"][s] = run_algorithm("AW-NSGA-II", sc, s).best_assign

    # ---- phase 1: agnostic runs ------------------------------------------
    raw: list[dict] = []
    for kind in ("congestion", "fault", "urgent"):
        for k, inst in enumerate(make_instances(kind)):
            inst = dict(inst, _idx=k)
            for s in seeds:
                variants = ["agnostic"] if kind != "fault" \
                    else ["agnostic", "agnostic-excl"]
                for v in variants:
                    raw.append(run_agnostic(kind, s, inst, plans, v))
            if verbose:
                print(f"  [{kind}] instance {k}: done", flush=True)

    _write_csv(os.path.join(RESULTS, "e_c17_raw.csv"), raw)

    # ---- aggregation ------------------------------------------------------
    def _mstats(vals):
        vals = np.asarray(vals, dtype=float)
        return (float(vals.mean()),
                float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)

    agg_rows = []
    by = defaultdict(list)
    for r in raw:
        by[(r["scenario"], r["strategy"])].append(r)
    # include published EHCO / A*-GA for side-by-side aggregation
    for (scn, strat, _s, _i), pr in pub.items():
        if strat in ("EHCO", "A*-GA"):
            by[(scn, strat)].append({k: (v if k != "timeout" else int(v))
                                     for k, v in pr.items()})
    for (scn, strat), rows in sorted(by.items()):
        m, s_ = _mstats([float(r["deg_pct"]) for r in rows])
        agg_rows.append({"scenario": scn, "strategy": strat,
                         "metric": "deg_pct", "mean": m, "std": s_,
                         "n": len(rows)})
        to = [int(r["timeout"]) for r in rows if r.get("timeout") != ""]
        if to:
            agg_rows.append({"scenario": scn, "strategy": strat,
                             "metric": "timeout_rate", "mean": float(np.mean(to)),
                             "std": 0.0, "n": len(to)})
        done = [r for r in rows if int(r.get("timeout") or 0) == 0]
        if done:
            m2, s2 = _mstats([float(r["deg_pct"]) for r in done])
            agg_rows.append({"scenario": scn, "strategy": strat,
                             "metric": "deg_pct_completed", "mean": m2,
                             "std": s2, "n": len(done)})
        resp = [float(r["response_ms"]) for r in rows if r.get("response_ms")
                not in ("", None)]
        if resp:
            m3, s3 = _mstats(resp)
            agg_rows.append({"scenario": scn, "strategy": strat,
                             "metric": "response_ms", "mean": m3, "std": s3,
                             "n": len(resp)})
    _write_csv(os.path.join(RESULTS, "e_c17_agg.csv"), agg_rows)

    # ---- paired statistics -------------------------------------------------
    def _pair(scn, strat_a, strat_b):
        a, b = [], []
        for r in raw:
            if r["scenario"] != scn or r["strategy"] != strat_b:
                continue
            pa = pub.get((scn, strat_a, r["seed"], r["instance"]))
            if pa is None:
                continue
            a.append(float(pa["deg_pct"]))
            b.append(float(r["deg_pct"]))
        return np.array(a), np.array(b)

    stat_rows = []
    COMPARISONS = [
        ("congestion", "EHCO", "agnostic"),
        ("urgent", "EHCO", "agnostic"),
        ("fault", "EHCO", "agnostic"),
        ("fault", "EHCO", "agnostic-excl"),
        ("fault", "A*-GA", "agnostic"),          # replanning value alone
        ("fault", "A*-GA", "agnostic-excl"),
    ]
    for scn, sa, sb in COMPARISONS:
        a, b = _pair(scn, sa, sb)
        if len(a) < 5:
            continue
        p = paired_wilcoxon(b, a, alternative="two-sided")
        stat_rows.append({"scenario": scn, "strategy_a": sa, "strategy_b": sb,
                          "metric": "deg_pct", "mean_a": float(a.mean()),
                          "mean_b": float(b.mean()), "n": len(a),
                          "wilcoxon_p": p, "cohen_dz": cohen_dz(b, a)})
    _write_csv(os.path.join(RESULTS, "e_c17_stats.csv"), stat_rows)

    meta = {
        "seeds": seeds, "variants": ["agnostic", "agnostic-excl"],
        "agnostic_objective": "(f1,f2,f3), crowding NSGA-II, no FLC, no f4, no c_m",
        "budgets": {"local": C.NP_LOCAL * C.GMAX_LOCAL,
                    "global": C.NP_GLOBAL * C.GMAX_GLOBAL},
        "initial_plan": "published AW-NSGA-II (identical to E4 EHCO rows)",
        "published_e4": os.path.relpath(PUBLISHED_E4),
        "trigger": "identical to EHCO (congestion onset / urgent insertion / "
                   "platform health warning h_cum < 0.6)",
    }
    with open(os.path.join(RESULTS, "e_c17_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    if verbose:
        print(f"[C17] total wall: {time.perf_counter()-t0:.1f}s; "
              f"{len(raw)} new runs", flush=True)
    return {"n_seeds": len(seeds), "n_runs": len(raw), "out_dir": RESULTS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="comma list, e.g. 1,2,3")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    run_c17(seeds=seeds)


if __name__ == "__main__":
    main()
