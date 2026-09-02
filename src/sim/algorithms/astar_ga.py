"""A*-GA baseline (SPEC.md section 8): single-objective GA on f1 (makespan).

External baseline from the manuscript: routing via plain A* (standard distance
tables, no improved A*), no health objective, no maintenance cost, no health
hard constraint. Same D3 operators and equal 20,000-eval budget.
"""
from __future__ import annotations

from ..objectives import EvalConfig
from ..scenario import Scenario
from .common import AlgorithmResult, EvalBudget, nsga2_core

NAME = "A*-GA"


def run(sc: Scenario, seed: int, cfg: EvalConfig | None = None,
        budget: EvalBudget | None = None) -> AlgorithmResult:
    if cfg is None:
        cfg = EvalConfig(use_health_objective=False,
                         use_maintenance_cost=False,
                         use_improved_astar=False)
    return nsga2_core(
        sc, seed, obj_idx=(0,), cfg=cfg, budget=budget,
        name=NAME, use_crowding=False,
        params_extra={"last_front_fill": "f1-only"})
