# -*- coding: utf-8 -*-
"""E-fig6 v3: exact reproduction of e1 + per-generation HV curve.

For each seed (1..30), ALL EIGHT algorithms of results/e1/e1_main.py are
rerun with default arguments (bit-identical to the E1 runs).  The union-of-8
normalisation basis is rebuilt exactly as e1's phase 2 (_front4 of every
algorithm, full 4-D non-dominated filter of F_final), and the final HV of
every (algo, seed) is verified against e1_metrics.csv 'hv' (HV-E1).

For AW-NSGA-II / MOEA/D / NSGA-III, the convergence curve is
  (g, HV) for g in front_history   [population after selection, every 10 gens]
plus one final point (gens_actual, HV of _front4) which equals HV-E1 exactly.

Output: sim/results/e_fig6/hv_convergence.csv
        (algorithm, generation, mean, std, n)
"""
import os, sys, csv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from sim.scenario import make_scenario
from sim.algorithms import ORDER, run_algorithm
from sim.algorithms.common import finite_F
from sim.metrics import fast_non_dominated_sort, hypervolume

OUT_DIR = os.path.join(ROOT, "sim", "results", "e_fig6")
SEEDS = list(range(1, 31))
PLOT_ALGOS = ["AW-NSGA-II", "MOEA/D", "NSGA-III"]
REF = np.ones(4)


def _front4(res):
    """e1_main._front4: full 4-D non-dominated filter of F_final."""
    idx = fast_non_dominated_sort(finite_F(res.F_final))[0]
    return res.F_final[idx]


def _hv_union(F, lo, span):
    Fn = (F - lo) / span
    return hypervolume(Fn, REF)


# ---- 1) reference HV-E1 per (algo, seed) from e1_metrics.csv --------------
hv_e1 = {}
with open(os.path.join(ROOT, "sim", "results", "e1", "e1_metrics.csv"),
          encoding="utf-8") as f:
    for r in csv.DictReader(f):
        hv_e1[(r["algo"], int(r["seed"]))] = float(r["hv"])

# ---- 2) rerun all 8 algorithms per seed; rebuild union4 exactly as e1 -----
all_curves = {a: {} for a in PLOT_ALGOS}   # algo -> seed -> [(gen, hv)]
n_verified = 0
n_total = 0
for seed in SEEDS:
    sc = make_scenario(seed)
    res_all = {}
    for name in ORDER:
        res_all[name] = run_algorithm(name, sc, seed)   # e1 default call
    union4 = np.vstack([_front4(res_all[name]) for name in ORDER])
    lo = union4.min(axis=0)
    hi = union4.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)

    # verify every algorithm's final HV against e1_metrics (exact D5 basis)
    for name in ORDER:
        hv_fin = _hv_union(_front4(res_all[name]), lo, span)
        target = hv_e1[(name, seed)]
        n_total += 1
        if abs(hv_fin - target) <= 1e-9:
            n_verified += 1
        else:
            print("MISMATCH %s seed %d: %.9f vs e1 %.9f (diff %.2e)"
                  % (name, seed, hv_fin, target, hv_fin - target))

    # convergence curves for the three plotted algorithms
    for algo in PLOT_ALGOS:
        res = res_all[algo]
        curve = []
        for g, F in res.front_history:
            F4 = np.asarray(F, dtype=float)[:, :4]
            curve.append((int(g), _hv_union(F4, lo, span)))
        hv_final = _hv_union(_front4(res), lo, span)
        g_final = int(res.params.get("gens_actual", 200))
        if curve and curve[-1][0] >= g_final:
            curve[-1] = (g_final, hv_final)
        else:
            curve.append((g_final, hv_final))
        all_curves[algo][seed] = curve
    print("seed %d done (union4 %d pts)" % (seed, len(union4)), flush=True)

print("verification: %d/%d (algo, seed) final HVs match e1 HV-E1 (<=1e-9)"
      % (n_verified, n_total))

# ---- 3) aggregate ----------------------------------------------------------
gens = sorted({g for a in PLOT_ALGOS for c in all_curves[a].values() for g, _ in c})
rows = []
for algo in PLOT_ALGOS:
    for g in gens:
        vals = [dict(c)[g] for c in all_curves[algo].values() if g in dict(c)]
        if vals:
            rows.append({"algorithm": algo, "generation": g,
                         "mean": round(float(np.mean(vals)), 6),
                         "std": round(float(np.std(vals, ddof=1)), 6),
                         "n": len(vals)})

os.makedirs(OUT_DIR, exist_ok=True)
path = os.path.join(OUT_DIR, "hv_convergence.csv")
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["algorithm", "generation", "mean", "std", "n"])
    w.writeheader()
    w.writerows(rows)
print("written", path)

g_max = max(gens)
print("\nfinal-generation HV-E1 means (must match Table 3):")
for algo in PLOT_ALGOS:
    for r in rows:
        if r["algorithm"] == algo and r["generation"] == g_max:
            print("  %-12s gen %d: %.3f +/- %.3f (n=%d)"
                  % (algo, r["generation"], r["mean"], r["std"], r["n"]))
print("E-fig6 v3 complete.")
