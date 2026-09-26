"""Temp: E1 union-normalised HV for MOEA/D vs AW-FLC+crowd pm=0.01."""
import numpy as np
from sim import constants as C
from sim.scenario import make_scenario
from sim.algorithms import moead
from sim.algorithms.common import EvalBudget, nsga2_core, flc_weight_provider, finite_F
from sim.metrics import fast_non_dominated_sort, hypervolume

seeds = list(range(1, 31))
orig = (C.PM_MIN, C.PM_MAX)
C.PM_MIN = C.PM_MAX = 0.01

def _front(res):
    idx = fast_non_dominated_sort(finite_F(res.F_final))[0]
    return res.F_final[idx]

def aw_crowd(sc, s):
    return nsga2_core(sc, s, obj_idx=(0, 1, 2, 3), cfg=None, budget=EvalBudget(),
                      name="AW-crowd", weight_fn=flc_weight_provider, use_crowding=True)

# collect per-seed union of the two fronts
all_fronts = {}
for s in seeds:
    sc = make_scenario(s)
    U = np.vstack([_front(moead.run(sc, s)), _front(aw_crowd(sc, s))])
    all_fronts[s] = U
    print(f"  seed {s:3d} union |F|={len(U)}", flush=True)

def union_hv(fn, s):
    F = _front(fn(make_scenario(s), s))
    U = all_fronts[s]
    lo, hi = U.min(axis=0), U.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    Fn = (F - lo) / span
    return hypervolume(Fn, np.ones(4))

rows_m, rows_a = [], []
for s in seeds:
    rows_m.append(union_hv(moead.run, s))
    rows_a.append(union_hv(aw_crowd, s))
a = np.array(rows_m); b = np.array(rows_a)
print("--- union-HV (per-seed union of MOEA/D + AW-crowd) ---")
print(f"MOEA/D       union-HV={a.mean():.4f}+-{a.std():.4f}")
print(f"AW-crowd0.01 union-HV={b.mean():.4f}+-{b.std():.4f}")
C.PM_MIN, C.PM_MAX = orig
print("ALL DONE")
