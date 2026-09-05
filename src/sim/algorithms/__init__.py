"""Algorithm registry (SPEC.md section 8): eight configurations, one entry point.

All algorithms consume the same Scenario, use equal 20,000-eval budgets, and
return a uniform :class:`AlgorithmResult`. Names used by the experiments layer
and result tables.

EvalConfig defaults per algorithm:
- AW-NSGA-II / NSGA-II-2obj / NSGA-II-3obj / NSGA-III / MOEA/D / LWC : default
  full model (f1-f4, maintenance cost, hard h>=h_crit constraint).
- A*-GA  : no health objective / no maintenance cost / standard A* tables.
- GA-SA  : hard exclusion at h < 0.6 (battery-SoH proxy), weighted scalar.
"""
from __future__ import annotations

from ..objectives import EvalConfig
from ..scenario import Scenario
from . import astar_ga, aw_nsga2, ga_sa, lwc, moead, nsga2, nsga3
from .common import AlgorithmResult, EvalBudget

REGISTRY = {
    "AW-NSGA-II": aw_nsga2.run,
    "NSGA-II-2obj": nsga2.run_2obj,
    "NSGA-II-3obj": nsga2.run_3obj,
    "NSGA-III": nsga3.run,
    "MOEA/D": moead.run,
    "GA-SA": ga_sa.run,
    "A*-GA": astar_ga.run,
    "LWC": lwc.run,
}

ORDER = list(REGISTRY)                     # canonical table order (SPEC 8)


def run_algorithm(name: str, sc: Scenario, seed: int,
                  cfg: EvalConfig | None = None,
                  budget: EvalBudget | None = None,
                  **kwargs) -> AlgorithmResult:
    """Run one algorithm on a scenario seed; raises KeyError for bad names.

    Extra keyword arguments (e.g. tracker for anytime studies) are forwarded
    to the algorithm entry point; algorithms that do not declare them raise a
    TypeError, so callers must only pass kwargs the target supports.
    """
    if name not in REGISTRY:
        raise KeyError(f"unknown algorithm {name!r}; available: {ORDER}")
    return REGISTRY[name](sc, seed, cfg, budget, **kwargs)


__all__ = ["REGISTRY", "ORDER", "run_algorithm", "AlgorithmResult",
           "EvalBudget", "aw_nsga2", "nsga2", "nsga3", "moead", "ga_sa",
           "astar_ga", "lwc"]
