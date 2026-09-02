"""C19: dynamic MOEA/D closed-loop baseline (expert pre-review, gap #1).

Motivation
----------
The published E4 dynamic experiments compare the EHCO closed loop only
against an open-loop A*-GA baseline, and the dynamic ablation (e4_ablation)
only removes FLC / adaptive weights *within the NSGA-II family*.  The expert
pre-review identified the missing rung as the most important experimental
gap: a closed-loop controller whose optimiser is MOEA/D -- the algorithm
that dominated the static E1 comparison (HV 0.765 vs 0.534) -- so that the
incremental value of the AW-NSGA-II replanning core can be attributed
against the strongest available optimiser rather than only against
"no replanning".

Controller variants (identical triggers, identical evaluation budgets,
identical incumbent guard and congestion-aware distance inflation as EHCO):
  moead-cl   : published AW-NSGA-II initial plan + MOEA/D replanning.
               Isolates the REPLANNING OPTIMISER (only the closed-loop
               policy changes -- the C17 attribution pattern).
  moead-full : MOEA/D initial plan + MOEA/D replanning.  The full MOEA/D
               closed loop a practitioner would deploy; answers "why not
               simply use MOEA/D for everything?".

MOEA/D runs its native configuration (100 Dirichlet weight vectors, fixed
config seed, Tchebycheff decomposition, neighbourhood T=20) under the SAME
evaluation budgets as the EHCO replanning calls (50 x 80 local, 100 x 200
global).  MOEA/D optimises the same four objectives (f1..f4) with the same
EvalConfig, so the health-aware feasibility repair is preserved.

Result ladder per scenario (deg_pct and f1_event, paired over seed+instance):
  A*-GA (open loop, published)  ->  noFLC (closed-loop NSGA-II, published)
  ->  moead-cl  ->  moead-full  ->  EHCO (published)

Outputs (sim/results/e_c19/):
  e_c19_raw.csv        all new rows (same schema as e4_raw.csv + variant)
  e_c19_agg.csv        mean +- std deg_pct / f1_event / response per strategy
  e_c19_stats.csv      paired two-sided Wilcoxon vs EHCO and vs A*-GA
  e_c19_meta.json      protocol dump

Run from the project root:  python -m sim.experiments.e_c19 [--variants ...]
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
from ..algorithms.common import EvalBudget
from ..algorithms.moead import run as moead_run
from ..experiments.e4_dynamic import (DynamicSimulator, SEEDS, make_instances)
from ..metrics import cohen_dz, paired_wilcoxon
from ..scenario import make_scenario

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "e_c19")
PUBLISHED_E4 = os.path.join(os.path.dirname(__file__), "..", "..",
                            "results", "e4", "e4_raw.csv")
PUBLISHED_ABL = os.path.join(os.path.dirname(__file__), "..", "..",
                             "results", "e4", "e4_ablation_raw.csv")

VARIANTS = ("moead-cl", "moead-full")
SCENARIOS = ("congestion", "fault", "urgent")


# --------------------------------------------------------------------------
# MOEA/D closed-loop controller
# --------------------------------------------------------------------------

class MoeadSimulator(DynamicSimulator):
    """Closed-loop controller whose replanning optimiser is MOEA/D.

    Everything else -- event triggers, replanning budgets, incumbent
    adoption guard, congestion-aware distance inflation, faulty-AGV
    exclusion and maintenance travel -- is inherited unchanged from the
    published EHCO path.  Only the optimiser core inside ``_replan`` is
    swapped: ``nsga2_core`` -> ``moead.run`` (native 100-weight-vector
    configuration, same EvalBudget)."""

    def _replan(self, sc_p, rng, pop_size, budget, name,
                incumbent=None):
        if len(sc_p.tasks) == 0 or len(sc_p.agvs) == 0:
            self.replan_info.append({"name": name, "budget": 0,
                                     "pop": pop_size,
                                     "n_tasks": len(sc_p.tasks),
                                     "n_agv": len(sc_p.agvs),
                                     "wall_ms": 0.0, "evals": 0,
                                     "skipped": True})
            return 0.0
        t0 = time.perf_counter()
        res = moead_run(sc_p, seed=int(rng.integers(1, 2 ** 31)),
                        cfg=self.cfg, budget=EvalBudget(budget))
        wall_ms = (time.perf_counter() - t0) * 1000.0
        self.replan_walls_ms.append(wall_ms)
        info = {"name": name + "-moead", "budget": budget, "pop": 100,
                "n_tasks": len(sc_p.tasks), "n_agv": len(sc_p.agvs),
                "wall_ms": wall_ms, "evals": res.evals}
        applied = True
        if incumbent is not None:
            cur = self._eval_incumbent(incumbent, sc_p)
            f1_cur = float(cur["f1"][0])
            f1_new = float(res.best_F[0])
            applied = bool(f1_new <= f1_cur)
            info.update({"incumbent_f1": f1_cur, "new_f1": f1_new,
                         "applied": applied})
        self.replan_info.append(info)
        if applied:
            self._apply_new_assign(res.best_assign, sc_p)
        return wall_ms

    def _eval_incumbent(self, incumbent, sc_p):
        # local import to avoid a cycle at module load time
        from ..objectives import evaluate_population
        return evaluate_population(incumbent.reshape(1, -1), sc_p, self.cfg)


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def run_moead_variant(kind: str, seed: int, inst: dict, plans: dict,
                      variant: str) -> dict:
    """One (seed, instance, moead-variant) run; schema mirrors e4_raw.csv."""
    sc = plans["scenarios"][seed]
    assign = plans["aw"][seed] if variant == "moead-cl" \
        else plans["moead"][seed]
    ev = dict(inst)
    ev["rng"] = np.random.default_rng(seed * 1000 + 777)

    sim = MoeadSimulator(sc, assign)
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


def _load_published(path: str, strategies: tuple[str, ...]) -> dict[tuple, dict]:
    pub: dict[tuple, dict] = {}
    if not os.path.exists(path):
        return pub
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["strategy"] in strategies:
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


def run_c19(seeds: list[int] | None = None,
            scenarios: list[str] | None = None,
            variants: list[str] | None = None,
            verbose: bool = True) -> dict:
    seeds = seeds or SEEDS
    scenarios = scenarios or list(SCENARIOS)
    variants = variants or list(VARIANTS)
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.perf_counter()

    # ---- phase 0: initial plans -------------------------------------------
    if verbose:
        print(f"[C19] phase 0: initial plans ({len(seeds)} seeds)...",
              flush=True)
    plans: dict = {"aw": {}, "moead": {}, "scenarios": {}}
    need_moead_plan = any(v == "moead-full" for v in variants)
    e1_f1: dict[int, float] = {}
    e1_path = os.path.join(os.path.dirname(RESULTS), "e1", "e1_metrics.csv")
    if os.path.exists(e1_path):
        with open(e1_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["algo"] == "AW-NSGA-II":
                    e1_f1[int(r["seed"])] = float(r["f1"])
    aw_check = []
    for s in seeds:
        sc = make_scenario(s)
        plans["scenarios"][s] = sc
        res_aw = run_algorithm("AW-NSGA-II", sc, s)
        plans["aw"][s] = res_aw.best_assign
        aw_check.append(res_aw.best_F[0])
        if need_moead_plan:
            plans["moead"][s] = run_algorithm("MOEA/D", sc, s).best_assign
        if verbose:
            print(f"    seed {s:3d} AW f1={res_aw.best_F[0]:7.1f}", flush=True)
    bad = [(s, v) for s, v in zip(seeds, aw_check)
           if s in e1_f1 and abs(v - e1_f1[s]) > 1e-6]
    print("[C19] AW initial-plan check vs e1_metrics.csv: "
          + ("exact match" if not bad else f"WARNING mismatch: {bad[:5]}"),
          flush=True)

    # ---- phase 1: moead closed-loop runs ------------------------------------
    raw: list[dict] = []
    for kind in scenarios:
        for k, inst in enumerate(make_instances(kind)):
            inst = dict(inst, _idx=k)
            for s in seeds:
                for v in variants:
                    raw.append(run_moead_variant(kind, s, inst, plans, v))
            if verbose:
                print(f"  [{kind}] instance {k}: done", flush=True)

    _write_csv(os.path.join(RESULTS, "e_c19_raw.csv"), raw)

    # ---- phase 2: aggregation + paired stats --------------------------------
    pub = _load_published(PUBLISHED_E4, ("EHCO", "A*-GA"))
    abl = _load_published(PUBLISHED_ABL, ("noFLC",))

    def _mstats(vals):
        vals = np.asarray(vals, dtype=float)
        return (float(vals.mean()),
                float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)

    agg_rows = []
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in raw:
        by[(r["scenario"], r["strategy"])].append(r)
    for (scn, strat, _s, _i), pr in pub.items():
        by[(scn, strat)].append(pr)
    for (scn, strat, _s, _i), pr in abl.items():
        by[(scn, strat)].append(pr)
    for (scn, strat), rows in sorted(by.items()):
        m, s_ = _mstats([float(r["deg_pct"]) for r in rows])
        agg_rows.append({"scenario": scn, "strategy": strat,
                         "metric": "deg_pct", "mean": m, "std": s_,
                         "n": len(rows)})
        m, s_ = _mstats([float(r["f1_event"]) for r in rows])
        agg_rows.append({"scenario": scn, "strategy": strat,
                         "metric": "f1_event", "mean": m, "std": s_,
                         "n": len(rows)})
        resp = [float(r["response_ms"]) for r in rows
                if r.get("response_ms") not in ("", None)]
        if resp:
            m, s_ = _mstats(resp)
            agg_rows.append({"scenario": scn, "strategy": strat,
                             "metric": "response_ms", "mean": m, "std": s_,
                             "n": len(resp)})
    _write_csv(os.path.join(RESULTS, "e_c19_agg.csv"), agg_rows)

    def _pair(scn, strat_a, strat_b, metric):
        """Pair published/reference rows (a) with new moead rows (b)."""
        a, b = [], []
        for r in raw:
            if r["scenario"] != scn or r["strategy"] != strat_b:
                continue
            pa = pub.get((scn, strat_a, r["seed"], r["instance"])) \
                or abl.get((scn, strat_a, r["seed"], r["instance"]))
            if pa is None:
                continue
            a.append(float(pa[metric]))
            b.append(float(r[metric]))
        return np.array(a), np.array(b)

    stat_rows = []
    for scn in scenarios:
        for v in variants:
            for ref in ("EHCO", "A*-GA", "noFLC"):
                for metric in ("deg_pct", "f1_event"):
                    a, b = _pair(scn, ref, v, metric)
                    if len(a) < 5:
                        continue
                    p = paired_wilcoxon(b, a, alternative="two-sided")
                    stat_rows.append({
                        "scenario": scn, "strategy_a": ref, "strategy_b": v,
                        "metric": metric, "mean_a": float(a.mean()),
                        "mean_b": float(b.mean()), "n": len(a),
                        "wilcoxon_p": p, "cohen_dz": cohen_dz(b, a)})
    _write_csv(os.path.join(RESULTS, "e_c19_stats.csv"), stat_rows)

    meta = {
        "seeds": seeds, "scenarios": scenarios, "variants": variants,
        "replan_optimiser": "MOEA/D (100 Dirichlet wv, seed 20260901, "
                            "Tchebycheff, T=20), same EvalBudget as EHCO",
        "initial_plan": {"moead-cl": "AW-NSGA-II (published)",
                         "moead-full": "MOEA/D"},
        "budgets": {"local": C.NP_LOCAL * C.GMAX_LOCAL,
                    "global": C.NP_GLOBAL * C.GMAX_GLOBAL},
        "published_e4": os.path.relpath(PUBLISHED_E4),
        "trigger": "identical to EHCO",
    }
    with open(os.path.join(RESULTS, "e_c19_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    if verbose:
        print(f"[C19] total wall: {time.perf_counter()-t0:.1f}s; "
              f"{len(raw)} new runs", flush=True)
    return {"n_seeds": len(seeds), "n_runs": len(raw), "out_dir": RESULTS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="comma list, e.g. 1,2,3")
    ap.add_argument("--scenarios", default=None,
                    help="comma list: congestion,fault,urgent")
    ap.add_argument("--variants", default=None,
                    help="comma list: moead-cl,moead-full")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    scenarios = args.scenarios.split(",") if args.scenarios else None
    variants = args.variants.split(",") if args.variants else None
    run_c19(seeds=seeds, scenarios=scenarios, variants=variants)


if __name__ == "__main__":
    main()
