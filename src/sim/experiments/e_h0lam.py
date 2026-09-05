# -*- coding: utf-8 -*-
"""E-H0LAM: h0-coupled true-degradation replay on the §3.5 ACTIVATION scenario.

Expert-opinion-10 attack #1 asks whether protecting the weak AGVs reduces their
PHYSICAL degradation once the degradation rate depends on the current health
state.  Table S10 already applies the h0 coupling (lambda in {0,1,2}) to the
mixed-health fleet (h0 ~ U[0.3, 1]); this module applies the same coupling to
the C16 activation scenario (five AGVs h0 ~ U[0.3, 0.6)) for the three arms
full / hard / nohealth, re-using the published optimisation solutions (the
decisions were taken under the published engine; only the physical replay is
re-computed under lambda coupling, exactly as in Table S10).

lambda semantics (objectives.simulate_plan_health h0_coupling): a vehicle of
initial health h0 < 1 wears faster — K_WEAR*(1 + lam*(1 - h0)) and thermal
multiplier (1 + 0.10*lam*(1 - h0)) — so weak vehicles degrade faster under the
same load when lam > 0.

Outputs (sim/results/e_h0lam/):
  c16_lambda_metrics.csv  per (arm, lam, seed): delta_h stats
  c16_lambda_agg.csv      per (arm, lam, metric) mean +- sd
  c16_lambda_vs.csv       paired (full vs nohealth / hard) per lam per metric
"""
import csv
import io
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from .. import constants as C
from ..metrics import gini
from ..objectives import build_plan, simulate_plan_health
from .e_c16 import make_c16_scenario, ARMS
from .e_c16 import arm_metrics  # noqa: F401 (unused; ensure c16 import side)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "e_h0lam")
LAMS = (0.0, 1.0, 2.0)
ARMS_RUN = ("full", "hard", "nohealth")
SEEDS_PATH = os.path.join(os.path.dirname(__file__), "..", "results",
                          "seeds.json")


def _replay_stats(res, sc, low, lam):
    plans = build_plan(res.best_assign, sc)
    sim = simulate_plan_health(plans, sc, h0_coupling=float(lam))
    low = np.asarray([i in low for i in range(len(sc.agvs))])
    dh = sim["delta_h"]
    h0 = np.array([a.h0 for a in sc.agvs])
    h_end = sim["h_end"]
    return {
        "delta_h_mean": float(dh.mean()),
        "delta_h_max": float(dh.max()),
        "gini_delta_h": float(gini(dh)),
        "delta_h_low_mean": float(dh[low].mean()),
        "delta_h_high_mean": float(dh[~low].mean()),
        "h_end_min_lowgrp": float(h_end[low].min()),
        "share_low": float(low[res.best_assign].mean()),
    }


def run(seeds=None, verbose=True):
    if seeds is None:
        if os.path.exists(SEEDS_PATH):
            import json
            with open(SEEDS_PATH, encoding="utf-8") as f:
                seeds = json.load(f)
        else:
            seeds = list(range(1, 31))
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.perf_counter()
    rows = []
    cached = {}
    for s in seeds:
        sc, low = make_c16_scenario(s)
        cached[s] = (sc, low, {a: ARMS[a](sc, s) for a in ARMS_RUN})
    for lam in LAMS:
        for s in seeds:
            sc, low, arms = cached[s]
            for a in ARMS_RUN:
                st = _replay_stats(arms[a], sc, low, lam)
                rows.append({"arm": a, "lam": lam, "seed": s, **st})
        if verbose:
            print(f"  lam={lam} done ({time.perf_counter()-t0:.0f}s)",
                  flush=True)

    def _w(path, out):
        if not out:
            return
        cols = list(out[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(cols) + "\n")
            for r in out:
                f.write(",".join(str(r[c]) for c in cols) + "\n")
        print(f"  wrote {path} ({len(out)} rows)", flush=True)

    _w(os.path.join(RESULTS, "c16_lambda_metrics.csv"), rows)

    by = defaultdict(list)
    for r in rows:
        for k in ("delta_h_mean", "delta_h_max", "gini_delta_h",
                  "delta_h_low_mean", "delta_h_high_mean",
                  "h_end_min_lowgrp"):
            by[(r["arm"], r["lam"], k)].append(float(r[k]))
    agg = []
    for (a, lam, k), v in by.items():
        v = np.asarray(v)
        agg.append({"arm": a, "lam": lam, "metric": k,
                    "mean": float(v.mean()), "std": float(v.std(ddof=1)),
                    "n": len(v)})
    _w(os.path.join(RESULTS, "c16_lambda_agg.csv"), agg)

    vs = []
    for lam in LAMS:
        f = [r for r in rows if r["arm"] == "full" and r["lam"] == lam]
        for ref in ("nohealth", "hard"):
            r2 = [r for r in rows if r["arm"] == ref and r["lam"] == lam]
            if len(f) != len(r2):
                continue
            for k in ("delta_h_low_mean", "gini_delta_h", "delta_h_mean"):
                a = np.asarray([float(x[k]) for x in f])
                b = np.asarray([float(x[k]) for x in r2])
                from ..metrics import paired_wilcoxon
                p = paired_wilcoxon(a, b, alternative="two-sided")
                vs.append({"lam": lam, "metric": k, "vs_arm": ref,
                           "mean_full": float(a.mean()),
                           "mean_ref": float(b.mean()), "p": p})
    _w(os.path.join(RESULTS, "c16_lambda_vs.csv"), vs)
    print(f"[E-H0LAM] done {time.perf_counter()-t0:.0f}s -> {RESULTS}",
          flush=True)
    return {"n_seeds": len(seeds)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    run(seeds=seeds)
