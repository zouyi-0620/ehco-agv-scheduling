"""E-fig6: HV 收敛曲线实验

对 AW-NSGA-II / MOEA/D / NSGA-III 各跑 30 种子，
利用 AlgorithmResult.front_history（每 10 代记录一次前沿），
计算固定参考归一化 HV（以该 run 最终前沿的 min-max 为归一化基），
输出每代 HV 均值 ± SD。

输出：sim/results/e_fig6/hv_convergence.csv
"""
import os, sys, csv, json, math
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.scenario import make_scenario
from sim.algorithms import run_algorithm, EvalBudget
from sim.objectives import EvalConfig
from sim.algorithms.common import AlgorithmResult, BUDGET_DEFAULT
from sim.metrics import hypervolume

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "sim", "results", "e_fig6")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = list(range(1, 31))
ALGOS = ["AW-NSGA-II", "MOEA/D", "NSGA-III"]

# Record per run: list of (gen, hv) tuples
all_runs = {a: [] for a in ALGOS}

for algo in ALGOS:
    for seed in SEEDS:
        sc = make_scenario(seed)
        budget = EvalBudget(limit=BUDGET_DEFAULT)
        res: AlgorithmResult = run_algorithm(algo, sc, seed, EvalConfig(), budget)
        # Normalisation bounds from final front (fixed for the run)
        F_final = res.F_final[:, list(res.obj_idx)]  # (P, d)
        mins = F_final.min(axis=0)
        maxs = F_final.max(axis=0)
        # Avoid division by zero
        rng = maxs - mins
        rng[rng == 0] = 1.0
        ref = np.ones(len(res.obj_idx))
        run_curve = []
        for g, F in res.front_history:
            Fsub = F[:, list(res.obj_idx)]
            Fnorm = (Fsub - mins) / rng
            # keep non-dominated (in case some are dominated)
            # simple: compute hv on all points (dominated points don't add volume)
            hv = hypervolume(Fnorm, ref)
            run_curve.append((g, hv))
        all_runs[algo].append(run_curve)
    print(f"{algo}: {len(all_runs[algo])} runs recorded")

# Align generations (all should be same: 0,10,20,...)
gens = sorted({g for curves in all_runs.values() for c in curves for g, _ in c})
print(f"generations: {gens[:5]}...{gens[-5:]}")

# Aggregate
rows = []
for algo in ALGOS:
    for g in gens:
        vals = []
        for curve in all_runs[algo]:
            # find hv at this gen (exact match expected)
            for gg, hv in curve:
                if gg == g:
                    vals.append(hv)
                    break
        if vals:
            rows.append({
                "algorithm": algo, "generation": g,
                "mean": round(float(np.mean(vals)), 6),
                "std": round(float(np.std(vals, ddof=1)), 6),
                "n": len(vals),
            })

path = os.path.join(OUT_DIR, "hv_convergence.csv")
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["algorithm", "generation", "mean", "std", "n"])
    w.writeheader()
    w.writerows(rows)
print(f"written {path}")
print("E-fig6 complete.")
