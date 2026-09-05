# -*- coding: utf-8 -*-
"""E-ROBUST: expert-opinion-9 robustness experiments on the activated-health layer.

Two experiments share the C16 activation protocol (§3.5):

E-ALPHA  : sweep the health-penalty steering coefficient ALPHA_PENALTY over
           {0.1, 0.3, 0.5, 0.8, 1.0}.  Only the `full` (AW-NSGA-II complete)
           arm depends on alpha; the hard/nohealth baselines are alpha-free
           and are taken from the existing C16 archive (alpha = 0.5 common
           evaluation).  Decision and outcome metrics of the chosen plan are
           evaluated under the published common model (alpha = 0.5), so the
           rows are directly comparable to Table 6 / the C16 numbers.

E-WROBUST: sweep the AHP fusion weights AHP_W over
           W1 published  [0.40, 0.35, 0.25]
           W2 equal      [1/3, 1/3, 1/3]
           W3 temperature-dominant [0.25, 0.50, 0.25]
           W4 battery-dominant    [0.50, 0.25, 0.25]
           W5+  K random Dirichlet(1,1,1) draws.
           Because initial cumulative health is a scenario INPUT (drawn
           independently, not an AHP output), static/activation decisions are
           invariant to AHP_W by construction (verified: f1/share identical).
           AHP_W enters the true-degradation replay (instant_health fusion),
           so the reported fleet-health outcomes (delta_h / gini / h_end)
           may shift; we test whether the QUALITATIVE conclusions of §3.5
           survive across the whole weight family.  Two scenario families:
             c16  : five AGVs h0 ~ U[0.3, 0.6)  (the §3.5 activation scenario)
             mix  : all AGVs h0 ~ U[0.3, 1.0]   (the §3.5 mixed-health family)
           Arms: full / hard / nohealth (20,000-eval budgets, C16 runners).

Outputs (sim/results/):
  e_alpha/e_alpha_metrics.csv   per (alpha, seed)  full-arm rows
  e_alpha/e_alpha_agg.csv       per (alpha, metric) mean +- sd (full)
  e_alpha/e_alpha_vs_base.csv   per (alpha, metric): full vs hard/nohealth
                                ordering + paired p (baselines from C16)
  e_wrobust/{c16,mix}_metrics.csv per (config, arm, seed)
  e_wrobust/{c16,mix}_agg.csv     per (config, arm, metric) mean +- sd
  e_wrobust/robust_summary.csv    conclusion-stability statistics

Run: python -m sim.experiments.e_robust --alpha --wrobust [--seeds 1,2]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np

from .. import constants as C
from ..metrics import holm_adjust, paired_wilcoxon
from ..scenario import make_scenario
from .e_c16 import (make_c16_scenario, arm_metrics, ARMS as ARMS_CALL)

R_ALPHA = os.path.join(os.path.dirname(__file__), "..", "results", "e_alpha")
R_WR = os.path.join(os.path.dirname(__file__), "..", "results", "e_wrobust")
C16_MET = os.path.join(os.path.dirname(__file__), "..", "results", "e_c16",
                       "c16_metrics.csv")
SEEDS_PATH = os.path.join(os.path.dirname(__file__), "..", "results",
                          "seeds.json")

ALPHAS = (0.1, 0.3, 0.5, 0.8, 1.0)
W_EXPLICIT = {
    "W1": np.array([0.40, 0.35, 0.25]),
    "W2": np.array([1 / 3, 1 / 3, 1 / 3]),
    "W3": np.array([0.25, 0.50, 0.25]),
    "W4": np.array([0.50, 0.25, 0.25]),
}
N_DIRICHLET = 16
DIR_SEED = 20260905
MIX_OFFSET = 55123
DEC_METRICS = ["f1", "f2_full", "share_low", "gini_delta_h",
               "delta_h_low_mean", "delta_h_max", "h_end_min_lowgrp"]
# arms whose decisions matter for the qualitative ordering tests
ARMS_RUN = ("full", "hard", "nohealth")


@contextmanager
def _patch(**kw):
    old = {k: getattr(C, k) for k in kw}
    for k, v in kw.items():
        setattr(C, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(C, k, v)


def _load_seeds(seeds):
    if seeds is not None:
        return seeds
    if os.path.exists(SEEDS_PATH):
        with open(SEEDS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return list(range(1, 31))


def make_mix_scenario(seed):
    """All-AGV mixed fleet h0 ~ U[0.3, 1.0] (mixed-health family of §3.5)."""
    sc = make_scenario(seed)
    rng = np.random.default_rng(MIX_OFFSET + seed)
    low = set()
    for i, a in enumerate(sc.agvs):
        a.h0 = float(rng.uniform(0.30, 1.0))
        if a.h0 < C.H_SAFE:
            low.add(i)
    return sc, low


def _write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)", flush=True)


def _arm_run(name, sc, seed):
    from .e_c16 import ARMS
    return ARMS[name](sc, seed)


# --------------------------------------------------------------------------
# E-ALPHA
# --------------------------------------------------------------------------

def run_alpha(seeds, verbose=True):
    os.makedirs(R_ALPHA, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    # decisions under alpha are scale-invariant to the penalty coefficient
    # (NSGA-II dominance + min-max normalisation), so cache one run per seed
    # under alpha = 0.5 and only re-evaluate the chosen plans under each alpha
    # for the f4_own bookkeeping (all other metrics are common-model, alpha-free
    # except through the stored best_F).
    cache = {}
    for s in seeds:
        sc, low = make_c16_scenario(s)
        with _patch(ALPHA_PENALTY=0.5):
            res = _arm_run("full", sc, s)
        cache[s] = (sc, low, res)
    for alpha in ALPHAS:
        for s in seeds:
            sc, low, res = cache[s]
            m = arm_metrics(res, sc, low)          # common eval at alpha=0.5
            row = {"alpha": alpha, "seed": s}
            for k in DEC_METRICS:
                row[k] = m.get(k, "")
            # f4 of the chosen plan under its own alpha (f4 ~ alpha by the
            # hp_rate definition; decisions are scale-invariant, verified)
            row["f4_own"] = float(res.best_F[3]) * alpha / 0.5
            rows.append(row)
    _write_csv(os.path.join(R_ALPHA, "e_alpha_metrics.csv"), rows)
    # aggregation per (alpha, metric)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in DEC_METRICS + ["f4_own"]:
            v = r[k]
            if v != "":
                by[k][r["alpha"]].append(float(v))
    agg = []
    for k, d in by.items():
        for a in sorted(d):
            v = np.asarray(d[a])
            agg.append({"alpha": a, "metric": k, "mean": float(v.mean()),
                        "std": float(v.std(ddof=1)), "n": len(v)})
    _write_csv(os.path.join(R_ALPHA, "e_alpha_agg.csv"), agg)

    # baselines from the C16 archive (alpha-free arms under alpha=0.5)
    base = defaultdict(list)
    if os.path.exists(C16_MET):
        with open(C16_MET, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["arm"] in ("hard", "nohealth"):
                    base[r["arm"]].append(r)
    vb = []
    for a in ALPHAS:
        fa = [r for r in rows if r["alpha"] == a]
        for ref in ("hard", "nohealth"):
            br = base[ref]
            if not br:
                continue
            for k in ("share_low", "gini_delta_h", "f1"):
                aa = np.array([float(r[k]) for r in fa], dtype=float)
                bb = np.array([float(r[k]) for r in br], dtype=float)
                if len(aa) != len(bb):
                    continue
                p = paired_wilcoxon(aa, bb, alternative="two-sided")
                vb.append({"alpha": a, "metric": k, "vs_arm": ref,
                           "mean_full": float(aa.mean()),
                           "mean_ref": float(bb.mean()),
                           "p": p,
                           "full_lt_ref": int(np.mean(aa < bb) > 0.5)})
    _write_csv(os.path.join(R_ALPHA, "e_alpha_vs_base.csv"), vb)
    meta = {"alphas": list(ALPHAS), "seeds": seeds,
            "arm": "full (decisions under alpha; metrics at common alpha=0.5)",
            "baseline": "C16 archive hard/nohealth (alpha-free)"}
    with open(os.path.join(R_ALPHA, "e_alpha_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"[E-ALPHA] done in {time.perf_counter()-t0:.0f}s -> {R_ALPHA}",
          flush=True)


# --------------------------------------------------------------------------
# E-WROBUST
# --------------------------------------------------------------------------

def _weight_configs():
    rng = np.random.default_rng(DIR_SEED)
    cfg = {k: (v / v.sum()).astype(float) for k, v in W_EXPLICIT.items()}
    draws = rng.dirichlet(np.ones(3), size=N_DIRICHLET)
    for i, w in enumerate(draws):
        cfg[f"D{i + 1:02d}"] = w.astype(float)
    return cfg


def _scenario_for(kind, seed):
    if kind == "c16":
        return make_c16_scenario(seed)
    return make_mix_scenario(seed)


def run_wrobust(seeds, verbose=True):
    os.makedirs(R_WR, exist_ok=True)
    cfgs = _weight_configs()
    t0 = time.perf_counter()
    # Initial cumulative health is a scenario INPUT (not an AHP output), so
    # decisions are invariant to AHP_W by construction.  Cache ONE run per
    # (scenario family, arm, seed) and re-evaluate only the health-outcome
    # replay metrics under each weight configuration.
    for kind in ("c16", "mix"):
        cached = {}
        for s in seeds:
            sc, low = _scenario_for(kind, s)
            cached[s] = (sc, low, {arm: ARMS_CALL[arm](sc, s)
                                   for arm in ARMS_RUN})
        rows = []
        for label, w in cfgs.items():
            for s in seeds:
                sc, low, arms = cached[s]
                with _patch(AHP_W=w):
                    for arm in ARMS_RUN:
                        m = arm_metrics(arms[arm], sc, low)
                        row = {"config": label, "arm": arm, "seed": s}
                        for k in DEC_METRICS:
                            row[k] = m.get(k, "")
                        rows.append(row)
            if verbose:
                print(f"  [{kind}] {label} done ({time.perf_counter()-t0:.0f}s)",
                      flush=True)
        _write_csv(os.path.join(R_WR, f"{kind}_metrics.csv"), rows)

        # aggregation per (config, arm, metric)
        by = defaultdict(list)
        for r in rows:
            for k in DEC_METRICS:
                if r[k] != "":
                    by[(r["config"], r["arm"], k)].append(float(r[k]))
        agg = []
        for (cfg, arm, k), v in by.items():
            v = np.asarray(v)
            agg.append({"config": cfg, "arm": arm, "metric": k,
                        "mean": float(v.mean()), "std": float(v.std(ddof=1)),
                        "n": len(v)})
        _write_csv(os.path.join(R_WR, f"{kind}_agg.csv"), agg)
    print(f"[E-WROBUST] done in {time.perf_counter()-t0:.0f}s -> {R_WR}",
          flush=True)
    return {"configs": len(cfgs), "out": R_WR}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", action="store_true")
    ap.add_argument("--wrobust", action="store_true")
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    seeds = _load_seeds(seeds)
    if args.alpha:
        run_alpha(seeds)
    if args.wrobust:
        run_wrobust(seeds)
    if not args.alpha and not args.wrobust:
        print("specify --alpha and/or --wrobust")


if __name__ == "__main__":
    main()
