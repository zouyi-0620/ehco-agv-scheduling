"""E-fig4: Task Load vs SoH 分层分析（补实验）

为 fig4 提供数据：对 3 个关键变体（Full=AW-NSGA-II, hard-only, GA-SA-multiSoH）
各跑 30 种子，记录 knee 解的每 AGV 任务分配数，按 SoH 分组，并计算 Spearman ρ。

输出：
  sim/results/e_fig4/taskload_per_run.csv   — 每 run 每 AGV 数据
  sim/results/e_fig4/taskload_summary.csv   — 按变体×SoH 分组的均值/标准差
  sim/results/e_fig4/spearman.csv           — 每 run 的 Spearman ρ
"""
import os, sys, csv, json, math
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.scenario import make_scenario
from sim.algorithms import run_algorithm, EvalBudget
from sim.objectives import EvalConfig, build_plan
from sim.algorithms.common import AlgorithmResult, EvalBudget, BUDGET_DEFAULT
from sim.constants import N_AGV

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "sim", "results", "e_fig4")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = list(range(1, 31))
VARIANTS = {
    "Full":            ("AW-NSGA-II", EvalConfig()),
    "hard-only":       ("AW-NSGA-II", EvalConfig(use_health_objective=False,
                                                   use_maintenance_cost=False,
                                                   hard_exclusion_only=True)),
    "GA-SA-multiSoH":  ("GA-SA", EvalConfig()),
}

def soh_band(soh: float) -> str:
    if soh < 0.80:
        return "low"
    elif soh < 0.90:
        return "mid"
    return "high"

per_run_rows = []
summary_rows = []
spearman_rows = []

for var_name, (algo_name, cfg) in VARIANTS.items():
    band_vals = {"low": [], "mid": [], "high": []}
    rho_vals = []
    for seed in SEEDS:
        sc = make_scenario(seed)
        budget = EvalBudget(limit=BUDGET_DEFAULT)
        res: AlgorithmResult = run_algorithm(algo_name, sc, seed, cfg, budget)
        # knee chromosome
        assign = res.best_assign
        if assign is None:
            # fallback: first non-dominated solution
            assign = res.population[0]
        # count tasks per AGV
        task_counts = np.zeros(N_AGV, dtype=int)
        for j in range(len(assign)):
            i = int(assign[j])
            task_counts[i] += 1
        sohs = np.array([a.h0 for a in sc.agvs])
        # per AGV row
        for i in range(N_AGV):
            per_run_rows.append({
                "variant": var_name, "seed": seed, "agv": i,
                "soh": round(sohs[i], 4), "task_count": int(task_counts[i]),
                "band": soh_band(sohs[i]),
            })
        # Spearman ρ (soh vs task_count across 10 AGVs)
        if len(set(task_counts)) > 1 and len(set(sohs)) > 1:
            rho, _ = stats.spearmanr(sohs, task_counts)
        else:
            rho = np.nan
        rho_vals.append(rho)
        # band means for this run
        for b in ("low", "mid", "high"):
            vals = [task_counts[i] for i in range(N_AGV) if soh_band(sohs[i]) == b]
            band_vals[b].append(np.mean(vals) if vals else np.nan)
    # summary per variant
    for b in ("low", "mid", "high"):
        arr = np.array([v for v in band_vals[b] if np.isfinite(v)], dtype=float)
        summary_rows.append({
            "variant": var_name, "band": b,
            "mean": round(float(np.mean(arr)), 3),
            "std": round(float(np.std(arr, ddof=1)), 3),
            "n": len(arr),
        })
    # spearman summary
    arr = np.array([v for v in rho_vals if np.isfinite(v)], dtype=float)
    spearman_rows.append({
        "variant": var_name,
        "rho_mean": round(float(np.mean(arr)), 3),
        "rho_std": round(float(np.std(arr, ddof=1)), 3),
        "n": len(arr),
    })
    print(f"{var_name}: rho={spearman_rows[-1]['rho_mean']:.3f}±{spearman_rows[-1]['rho_std']:.3f}  "
          f"loads low={summary_rows[-3]['mean']:.2f} mid={summary_rows[-2]['mean']:.2f} high={summary_rows[-1]['mean']:.2f}")

# write CSVs
for fname, rows, fieldnames in [
    ("taskload_per_run.csv", per_run_rows,
     ["variant", "seed", "agv", "soh", "task_count", "band"]),
    ("taskload_summary.csv", summary_rows,
     ["variant", "band", "mean", "std", "n"]),
    ("spearman.csv", spearman_rows,
     ["variant", "rho_mean", "rho_std", "n"]),
]:
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  written {path}")

print("E-fig4 complete.")
