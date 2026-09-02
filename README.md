# EHCO: Health-constrained Cooperative Path Optimization for Smart Warehousing

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22242890.svg)](https://doi.org/10.5281/zenodo.22242890)

Python reference implementation accompanying the manuscript

> **"Energy- and health-constrained cooperative multi-objective path optimization
> for smart warehouse AGV fleets"**
> Y. Zou, C. Mai, D. Wu — submitted to *Machines* (MDPI)

This repository contains the complete simulation and optimization framework used
to produce every experimental result in the manuscript: the static comparison
study (E1), the component ablations (E2), the dynamic fault/congestion event
study (E4, including the event-driven ablation isolating the FLC and
adaptive-weight contributions), and the parameter-sensitivity study (E6).

## Overview

EHCO couples an **AW-NSGA-II** optimizer (NSGA-II whose last-front selection is
driven by a fuzzy-logic-controller (FLC) adaptive weight vector) with a
multi-indicator **state-of-health (SoH)** model (battery cycle aging, thermal
dynamics, vibration wear) for an AGV fleet in a 50 × 40 m grid warehouse.
The FLC continuously re-weights the four objectives (makespan, energy,
distance/urgency balance, health) from live fleet state, and a closed-loop
replanning layer reacts to dynamic events (aisle congestion, equipment fault
warnings).

All experiments share:

- **30 fixed seeds** (1–30), identical Scenario instances per seed;
- an **equal 20,000-evaluation budget** per algorithm run (EvalBudget);
- paired **Wilcoxon signed-rank tests** with Holm–Bonferroni correction,
  Cohen's d_z effect sizes and 95% CIs for all pairwise comparisons.

## Requirements

- Python ≥ 3.10
- `numpy` (version-sensitive RNG: use **numpy 2.5.x**; the published numbers
  were generated on numpy 2.5.1)
- optional: `scipy`, `pandas` (statistics pipeline only)

```bash
pip install "numpy>=2.5"
```

## Reproducing the experiments

Run from the repository root (the package root is `src/`):

```bash
# E1: static main comparison (7 algorithms × 30 seeds, ~min/run)
python -m sim.experiments.e1_main

# E2: component ablations (Full / -health / -adaptW / -impAstar /
#      -closedLoop / hard-only / GA-SA-multiSoH)
python -m sim.experiments.e2_ablation

# E4: dynamic event study (aisle congestion + equipment fault warning,
#      with closed-loop replanning)
python -m sim.experiments.e4_dynamic

# E4 ablation: isolate FLC weight adaptation (-FLC) and the
#      adaptive-weight initial plan (-AW) under identical budgets
python -m sim.experiments.e4_ablation

# E6: one-at-a-time parameter sensitivity of AW-NSGA-II
python -m sim.experiments.e6_sensitivity
```

Each runner writes CSV/JSON outputs under `results/` with a fixed schema;
see the module docstrings for the exact output columns.

## Repository layout

```
src/sim/
  constants.py        single source of truth for all parameters (SPEC-derived)
  scenario.py         warehouse grid, task/fleet generation, scenario sampling
  warehouse.py        grid model, A* pathfinding, aisle capacity constraints
  dynamics.py         EWMA health estimator, thermal/vibration dynamics,
                      fault-injection protocol
  flc.py              fuzzy-logic controller (weight adaptation)
  objectives.py       f1–f4 objective functions, knee-point selection, EvalConfig
  metrics.py          HV, delta-h fleet statistics, convergence metrics
  algorithms/         NSGA-II / AW-NSGA-II / MOEA/D / NSGA-III / GA-SA /
                      improved-A*-GA / LWC baselines
  experiments/        E1 / E2 / E4 / E4-ablation / E6 runners + figure data
```

## Data availability

Generated datasets are available from the corresponding author upon
reasonable request. A Zenodo snapshot of this repository will be linked at
publication.

## License

MIT — see [LICENSE](LICENSE).
