"""AW-NSGA-II (EHCO): NSGA-II + FLC-adaptive weights (SPEC.md section 7).

The FLC maps (rho, gamma, h_bar) to weights (w1..w4); the weights are used for
the 变权标量化 last-front fill (keep the lowest s(x) = sum w_k f̃_k), replacing
the standard crowding-distance criterion.
"""
from __future__ import annotations

from ..objectives import EvalConfig
from ..scenario import Scenario
from .common import (AlgorithmResult, EvalBudget, flc_weight_provider,
                     nsga2_core)

NAME = "AW-NSGA-II"


def run(sc: Scenario, seed: int, cfg: EvalConfig | None = None,
        budget: EvalBudget | None = None) -> AlgorithmResult:
    return nsga2_core(
        sc, seed, obj_idx=(0, 1, 2, 3), cfg=cfg, budget=budget,
        name=NAME, weight_fn=flc_weight_provider, use_crowding=True,
        params_extra={"weight_source": "FLC",
                      "tournament": "FLC-weighted-scalar",
                      "last_front_fill": "crowding-distance"})
