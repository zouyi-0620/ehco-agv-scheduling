"""AW-NSGA-II (EHCO): NSGA-II + FLC-adaptive weights (SPEC.md section 7).

The FLC maps (rho, gamma, h_bar) to weights (w1..w4); the weights are used as
a binary-tournament tie-break via the weighted scalarization s(x)=sum w_k f_k.
Non-dominated sorting and the last-front fill remain standard NSGA-II
(crowding distance, descending) - see manuscript Algorithm 1, lines 9-15.
"""
from __future__ import annotations

from ..objectives import EvalConfig
from ..scenario import Scenario
from .common import (AlgorithmResult, EvalBudget, flc_weight_provider,
                     nsga2_core)

NAME = "AW-NSGA-II"


def run(sc: Scenario, seed: int, cfg: EvalConfig | None = None,
        budget: EvalBudget | None = None,
        tracker=None) -> AlgorithmResult:
    return nsga2_core(
        sc, seed, obj_idx=(0, 1, 2, 3), cfg=cfg, budget=budget,
        name=NAME, weight_fn=flc_weight_provider, use_crowding=True,
        params_extra={"weight_source": "FLC",
                      "tournament": "FLC-weighted-scalar",
                      "last_front_fill": "crowding-distance"},
        tracker=tracker)
