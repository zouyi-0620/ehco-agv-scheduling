"""NSGA-II baselines with 2 or 3 objectives (SPEC.md section 8).

2-obj : (f1, f2)
3-obj : (f1, f2, f3)
Both use the standard crowding-distance last-front fill and the full default
EvalConfig (hard constraint h >= h_crit still applies at scenario level).
"""
from __future__ import annotations

from ..objectives import EvalConfig
from ..scenario import Scenario
from .common import AlgorithmResult, EvalBudget, nsga2_core

NAME_2 = "NSGA-II-2obj"
NAME_3 = "NSGA-II-3obj"


def run_2obj(sc: Scenario, seed: int, cfg: EvalConfig | None = None,
             budget: EvalBudget | None = None) -> AlgorithmResult:
    return nsga2_core(
        sc, seed, obj_idx=(0, 1), cfg=cfg, budget=budget,
        name=NAME_2, use_crowding=True,
        params_extra={"last_front_fill": "crowding-distance"})


def run_3obj(sc: Scenario, seed: int, cfg: EvalConfig | None = None,
             budget: EvalBudget | None = None) -> AlgorithmResult:
    return nsga2_core(
        sc, seed, obj_idx=(0, 1, 2), cfg=cfg, budget=budget,
        name=NAME_3, use_crowding=True,
        params_extra={"last_front_fill": "crowding-distance"})


def run(sc: Scenario, seed: int, cfg: EvalConfig | None = None,
        budget: EvalBudget | None = None, obj_idx: tuple[int, ...] = (0, 1)
        ) -> AlgorithmResult:
    """Generic entry (used by the experiment framework for either variant)."""
    if obj_idx == (0, 1, 2):
        return run_3obj(sc, seed, cfg, budget)
    return run_2obj(sc, seed, cfg, budget)
