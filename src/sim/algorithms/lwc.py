"""LWC baseline: linear weight controller replacing the FLC (SPEC.md section 8).

Same AW-NSGA-II structure (变权标量化 last-front fill) but the weights are
w4 = a + d*(1 - h_bar) with a=0.34, d=0.40 and the remaining mass balanced
equally - the manuscript's control group isolating the FLC contribution.
"""
from __future__ import annotations

from ..objectives import EvalConfig
from ..scenario import Scenario
from .common import (AlgorithmResult, EvalBudget, lwc_weight_provider,
                     nsga2_core)

NAME = "LWC"


def run(sc: Scenario, seed: int, cfg: EvalConfig | None = None,
        budget: EvalBudget | None = None) -> AlgorithmResult:
    return nsga2_core(
        sc, seed, obj_idx=(0, 1, 2, 3), cfg=cfg, budget=budget,
        name=NAME, weight_fn=lwc_weight_provider, use_crowding=False,
        params_extra={"weight_source": "LWC", "last_front_fill": "weighted-scalar"})
