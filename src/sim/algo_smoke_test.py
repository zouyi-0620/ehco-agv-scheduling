"""Algorithm-layer smoke test: run all eight configurations on seed 1 and
validate the shared invariants (equal budget, non-empty fronts, sane runtime).

Also sanity-checks the metric primitives (exact HV, Das-Dennis counts, ND sort).

Run from the project root:  python -m sim.algo_smoke_test
"""
from __future__ import annotations

import sys
import time

import numpy as np

from . import constants as C
from .algorithms import ORDER, REGISTRY, run_algorithm
from .metrics import (crowding_distance, das_dennis_ref_points,
                      fast_non_dominated_sort, hypervolume)
from .scenario import make_scenario


def _check_metrics() -> None:
    # exact HV: single point (0.5, 0.5) vs ref (1,1) -> 0.25
    hv = hypervolume(np.array([[0.5, 0.5]]), np.array([1.0, 1.0]))
    assert abs(hv - 0.25) < 1e-9, f"HV single point = {hv}"
    # two non-dominated points -> 0.16
    hv2 = hypervolume(np.array([[0.5, 0.8], [0.8, 0.5]]), np.array([1.0, 1.0]))
    assert abs(hv2 - 0.16) < 1e-9, f"HV two points = {hv2}"
    # 4-D single point
    hv4 = hypervolume(np.full((1, 4), 0.5), np.ones(4))
    assert abs(hv4 - 0.5 ** 4) < 1e-9, f"HV 4D = {hv4}"

    # Das-Dennis H=5, m=4 -> 56 reference points, rows sum to 1, unique
    W = das_dennis_ref_points(5, 4)
    assert W.shape == (56, 4), f"Das-Dennis shape {W.shape}"
    assert np.allclose(W.sum(axis=1), 1.0)
    assert len(np.unique(W, axis=0)) == 56

    # ND sort on a small known set
    F = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [3.0, 3.0]])
    fronts = fast_non_dominated_sort(F)
    assert len(fronts) == 2
    assert set(fronts[0].tolist()) == {0, 1, 2}
    assert fronts[1].tolist() == [3]

    # crowding distance: extremes infinite
    cd = crowding_distance(F)
    assert np.isinf(cd[0]) and np.isinf(cd[1]) and np.isinf(cd[3])
    print("[ok] metrics primitives (HV / Das-Dennis / ND sort / crowding)")


def _run_all(seed: int = 1) -> dict:
    sc = make_scenario(seed)
    results = {}
    t0 = time.perf_counter()
    for name in ORDER:
        t = time.perf_counter()
        res = run_algorithm(name, sc, seed)
        dt = time.perf_counter() - t
        results[name] = (res, dt)
        ok_budget = res.evals == C.NP * C.GMAX
        ok_fronts = len(res.fronts) >= 1 and len(res.fronts[0]) >= 1
        assert ok_budget, f"{name}: evals {res.evals} != 20000"
        assert ok_fronts, f"{name}: empty final front"
        f1 = res.best_F[0]
        assert np.isfinite(f1), f"{name}: non-finite best f1"
        print(f"  {name:<14} evals={res.evals:5d}  wall={dt:6.2f}s  "
              f"|F1|={len(res.fronts[0]):3d}  best f1={f1:8.1f}s  "
              f"f2={res.best_F[1]:8.1f}  f3={res.best_F[2]:8.1f}  f4={res.best_F[3]:.4f}")
    total = time.perf_counter() - t0
    print(f"[ok] 8 algorithms on seed {seed}: {total:.1f}s total "
          f"({total / len(ORDER):.1f}s avg per algorithm)")
    return results


def main() -> int:
    print("== metrics primitives ==")
    _check_metrics()
    print("== algorithm runs (seed 1, budget 20,000 evals) ==")
    _run_all()
    print("ALL ALGORITHM SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
