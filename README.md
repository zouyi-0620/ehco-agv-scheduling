# EHCO: Health-constrained Cooperative Path Optimization for Smart Warehousing

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22242890.svg)](https://doi.org/10.5281/zenodo.22242890)

Python reference implementation accompanying the manuscript

> **"Energy- and health-constrained cooperative multi-objective path optimization
> for smart warehouse AGV fleets"**
> Y. Zou, C. Mai, D. Wu — submitted to *Machines* (MDPI)

This repository contains the complete simulation and optimization framework used
to produce every experimental result in the manuscript: the static comparison
study (E1), the component ablations (E2), the dynamic fault/congestion event
study including the compound congestion–fault–urgent cascade scenario (E4,
together with the event-driven ablation isolating the FLC and adaptive-weight
contributions), the parameter-sensitivity study (E6), and the seven validation
experiments (E-C16 … E-C22).

## Overview

EHCO couples an **AW-NSGA-II** optimizer (NSGA-II whose last-front selection is
driven by a fuzzy-logic-controller (FLC) adaptive weight vector) with a
multi-indicator **state-of-health (SoH)** model (battery cycle aging, thermal
dynamics, vibration wear) for an AGV fleet in a 50 × 40 m grid warehouse.
The FLC continuously re-weights the four objectives (makespan, energy,
distance/urgency balance, health) from live fleet state, and a closed-loop
replanning layer reacts to dynamic events (aisle congestion, equipment fault
warnings, urgent-order insertions).

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
#      with closed-loop replanning; --scenario compound adds the
#      congestion–fault–urgent cascade scenario D)
python -m sim.experiments.e4_dynamic

# E4 ablation: isolate FLC weight adaptation (-FLC) and the
#      adaptive-weight initial plan (-AW) under identical budgets
python -m sim.experiments.e4_ablation

# E6: one-at-a-time parameter sensitivity of AW-NSGA-II
python -m sim.experiments.e6_sensitivity

# Validation experiments:
# E-C16: health-dimension activation (f4 switched on)
python -m sim.experiments.e_c16

# E-C17: dynamic-gain attribution (replanning-architecture ablation)
python -m sim.experiments.e_c17

# E-C18: detection severity boundary sweep
python -m sim.experiments.e_c18

# E-C19: closed-loop MOEA/D optimizer baseline
python -m sim.experiments.e_c19

# E-C20: compound-event cascade scenario
python -m sim.experiments.e_c20

# E-C21: health-tracking rate (beta) sensitivity
python -m sim.experiments.e_c21

# E-C22: dynamic path-cost ablation (standard vs improved A*)
python -m sim.experiments.e_c22
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
  experiments/        E1 / E2 / E4 / E4-ablation / E6 runners and the
                      E-C16 … E-C22 validation-experiment runners
results/              aggregated tables and per-run data backing every figure
                      and table of the manuscript (see Data availability)
```

## Data availability

All aggregated result tables (CSV) supporting Figures 2–7 and Tables 1–11 and
S1–S7 of the manuscript, together with the per-run data of the dynamic-event
study (E4) and the seven validation experiments (E-C16 health-dimension
activation, E-C17 dynamic-gain attribution, E-C18 detection severity
boundary, E-C19 closed-loop MOEA/D optimizer baseline, E-C20 compound-event
cascade, E-C21 health-tracking-rate sensitivity, E-C22 dynamic path-cost
ablation), are included in the `results/` directory of this repository
(release v1.0.1). The seed manifest for all 30-seed experiments is
`results/seeds.json`. Raw simulation outputs beyond these files are available
from the corresponding author upon reasonable request.

## License

MIT — see [LICENSE](LICENSE).
