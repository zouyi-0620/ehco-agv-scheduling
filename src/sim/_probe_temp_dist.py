# -*- coding: utf-8 -*-
"""Record motor temperature distribution for one static E1 run (seed 1, AW-NSGA-II)."""
import sys, os
if __name__ == '__main__':
    __package__ = 'sim'
import numpy as np
from sim.algorithms import run_algorithm
from sim.objectives import build_plan
from sim.scenario import make_scenario
from sim import constants as C


def simulate_plan_health_trace(plans, sc, substep=1.0):
    """Return per-AGV temperature time series along executed plans."""
    N = len(sc.agvs)
    T = np.full(N, C.T_AMB)
    traces = [[] for _ in range(N)]
    for i in range(N):
        for seg in plans[i]:
            lr = {'empty': C.LOAD_RATE_EMPTY,
                  'loaded': C.LOAD_RATE_LOADED,
                  'idle': C.LOAD_RATE_IDLE}[seg.kind]
            t_ss = C.T_AMB + C.K_THERMAL * lr
            tau = C.TAU_THERMAL if t_ss > T[i] else C.TAU_COOL
            dist_rate = seg.dist / seg.duration if seg.duration > 0 else 0.0
            t_remaining = seg.duration
            while t_remaining > 1e-9:
                dt = min(substep, t_remaining)
                T[i] = t_ss + (T[i] - t_ss) * np.exp(-dt / tau)
                traces[i].append(T[i])
                t_remaining -= dt
    return [np.array(t) for t in traces]


if __name__ == "__main__":
    sc = make_scenario(1)
    alg_res = run_algorithm('AW-NSGA-II', sc, 1)
    plans = build_plan(alg_res.best_assign, sc)
    traces = simulate_plan_health_trace(plans, sc, substep=1.0)
    all_t = np.concatenate(traces)
    print("samples:", len(all_t))
    print("min/mean/median/95%/max:", all_t.min(), all_t.mean(), np.median(all_t),
          np.percentile(all_t, 95), all_t.max())
    print("% time <40C:", (all_t < 40).mean() * 100)
    print("% time in [40,85]C:", ((all_t >= 40) & (all_t <= 85)).mean() * 100)
    print("% time >85C:", (all_t > 85).mean() * 100)
    print("% time >90C:", (all_t > 90).mean() * 100)
