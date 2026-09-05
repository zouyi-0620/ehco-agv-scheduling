# -*- coding: utf-8 -*-
"""E-fig6 v2: HV convergence under the Table-3 (HV-E1) union-of-8 normalisation.

For each seed, the min-max bounds are taken from the union of all eight
algorithms' final fronts (results/e1/e1_solutions.csv, identical points to
e1_main's D5 union4), then AW-NSGA-II / MOEA/D / NSGA-III are rerun with
front_history and their per-generation 4-D HV is computed under those shared
bounds with reference point (1,1,1,1).  Final-generation HV must reproduce
the per-seed HV-E1 values of e1_metrics.csv (verification below).

Output: sim/results/e_fig6/hv_convergence.csv (same schema as before).
"""
import os, sys, csv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from sim.scenario import make_scenario
from sim.algorithms import run_algorithm, EvalBudget
from sim.objectives import EvalConfig
from sim.algorithms.common import BUDGET_DEFAULT
from sim.metrics import hypervolume

OUT_DIR = os.path.join(ROOT, "sim", "results", "e_fig6")
SEEDS = list(range(1, 31))
ALGOS = ["AW-NSGA-II", "MOEA/D", "NSGA-III"]

# ---- 1) union-of-8 bounds per seed from e1_solutions.csv ------------------
bounds = {}
with open(os.path.join(ROOT, "sim", "results", "e1", "e1_solutions.csv"),
          encoding="utf-8") as f:
    for r in csv.DictReader(f):
        s = int(r["seed"])
        pt = [float(r["f1"]), float(r["f2"]), float(r["f3"]), float(r["f4"])]
        bounds.setdefault(s, []).append(pt)
for s in bounds:
    A = np.asarray(bounds[s])
    lo, hi = A.min(axis=0), A.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    bounds[s] = (lo, span)

# ---- 2) reference HV-E1 per (algo, seed) from e1_metrics.csv --------------
hv_e1 = {}
with open(os.path.join(ROOT, "sim", "results", "e1", "e1_metrics.csv"),
          encoding="utf-8") as f:
    for r in csv.DictReader(f):
        hv_e1[(r["algo"], int(r["seed"]))] = float(r["hv"])

# ---- 3) rerun the three algorithms with front_history ---------------------
all_runs = {a: {} for a in ALGOS}   # algo -> seed -> [(gen, hv)]
for algo in ALGOS:
    for seed in SEEDS:
        sc = make_scenario(seed)
        budget = EvalBudget(limit=BUDGET_DEFAULT)
        res = run_algorithm(algo, sc, seed, EvalConfig(), budget)
        lo, span = bounds[seed]
        ref = np.ones(4)
        curve = []
        for g, F in res.front_history:
            F4 = np.asarray(F, dtype=float)[:, :4]
            Fn = (F4 - lo) / span
            hv = hypervolume(Fn, ref)
            curve.append((g, hv))
        all_runs[algo][seed] = curve
        # verification: final recorded generation vs e1 HV-E1
        g_fin, hv_fin = curve[-1]
        target = hv_e1[(algo, seed)]
        if abs(hv_fin - target) > 1e-6:
            print("MISMATCH %s seed %d: final %.6f vs e1 HV-E1 %.6f (diff %.2e)"
                  % (algo, seed, hv_fin, target, hv_fin - target))
    print("%s done (%d seeds)" % (algo, len(all_runs[algo])))

# ---- 4) aggregate ----------------------------------------------------------
gens = sorted({g for a in ALGOS for c in all_runs[a].values() for g, _ in c})
rows = []
for algo in ALGOS:
    for g in gens:
        vals = [dict(c)[g] for c in all_runs[algo].values() if g in dict(c)]
        if vals:
            rows.append({"algorithm": algo, "generation": g,
                         "mean": round(float(np.mean(vals)), 6),
                         "std": round(float(np.std(vals, ddof=1)), 6),
                         "n": len(vals)})

path = os.path.join(OUT_DIR, "hv_convergence.csv")
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["algorithm", "generation", "mean", "std", "n"])
    w.writeheader()
    w.writerows(rows)
print("written", path)

# summary of final values
print("\nfinal-generation HV-E1 means (should match Table 3):")
for algo in ALGOS:
    rs = [r for r in rows if r["algorithm"] == algo and r["generation"] == max(gens)]
    for r in rs:
        print("  %-12s gen %d: %.3f +/- %.3f (n=%d)"
              % (algo, r["generation"], r["mean"], r["std"], r["n"]))
print("E-fig6 v2 complete.")
