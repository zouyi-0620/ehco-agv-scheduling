# EHCO: Equipment-Health-Constrained Closed-Loop Multi-Objective AGV Path Optimization

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22242889.svg)](https://doi.org/10.5281/zenodo.22242889)

Python reference implementation accompanying the manuscript

> **"Cooperative Multi-Objective AGV Path Optimization in Smart Warehousing:
> An IoT-Inspired Closed-Loop Design with an Equipment-Health Safety Margin"**
> Y. Zou, C. Mai, D. Wu — submitted to *Machines* (MDPI)

This repository contains the complete simulation and optimization framework
used to produce every experimental result in the manuscript: the static
comparison study (E1), the component ablations (E2), the dynamic
fault/congestion event study including the compound congestion–fault–urgent
cascade scenario (E4, together with the event-driven ablation isolating the
FLC and adaptive-weight contributions), the parameter-sensitivity study (E6),
the seven validation experiments (E-C16 … E-C22), the strict fixed-wall-clock
best-so-far study (E-B2), the activated-health robustness sweeps (E-ALPHA /
E-WROBUST) and the h0-coupled degradation replay (E-h0λ), plus independent
30-seed validation arms (seeds 31–60) for the headline protocols.

## Overview

EHCO couples an **AW-NSGA-II** optimizer (NSGA-II whose last-front selection
is driven by a fuzzy-logic-controller (FLC) adaptive weight vector) with a
multi-indicator **state-of-health (SoH)** model (battery cycle aging, thermal
dynamics, vibration wear) for an AGV fleet in a 50 × 40 m grid warehouse.
The FLC continuously re-weights the four objectives (makespan, energy,
distance/urgency balance, health) from live fleet state, and a closed-loop
replanning layer reacts to dynamic events (aisle congestion, equipment fault
warnings, urgent-order insertions).

All experiments share:

- **integer generator seeds 1–60** (identical Scenario instances per seed;
  1–30 = in-sample protocol, 31–60 = independent validation arms);
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
# E1: static main comparison (8 algorithms × 30 seeds)
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

# E-B2: strict fixed-wall-clock best-so-far (anytime) study
python -m sim.experiments.e_b2_anytime

# E-ROBUST: activated-health robustness — alpha sweep + AHP-weight family
python -m sim.experiments.e_robust

# E-h0λ: h0-coupled degradation replay (Table S15)
python -m sim.experiments.e_h0lam

# Validation experiments (main protocol, seeds 1–30):
python -m sim.experiments.e_c16   # health-dimension activation
python -m sim.experiments.e_c17   # dynamic-gain attribution
python -m sim.experiments.e_c18   # detection severity boundary sweep
python -m sim.experiments.e_c19   # closed-loop MOEA/D optimizer baseline
python -m sim.experiments.e_c20   # compound-event cascade scenario
python -m sim.experiments.e_c21   # health-tracking rate (beta) sensitivity
python -m sim.experiments.e_c22   # dynamic path-cost ablation

# Independent validation arms: run the same protocol on seeds 31–60
# (see module docstrings; outputs land under results/e1_indep/, e4_indep/,
#  e_c17_indep/, e_c19_indep/, e_c20_indep/, e4_ablation_indep/).
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
                      improved-A*-GA / LWC baselines (anytime tracker hooks
                      used by E-B2 are optional and inert by default)
  experiments/        E1 / E2 / E4 / E4-ablation / E6 / E-B2 / E-ROBUST /
                      E-h0λ runners and the E-C16 … E-C22 runners
results/              aggregated tables and per-run data backing every figure
                      and table of the manuscript (see Data availability)
```

## Data availability

All aggregated result tables (CSV) supporting Figures 2–8 and Tables 1–12 and
S1–S15 of the manuscript, together with the per-run data of the dynamic-event
study (E4) and its ablation, the seven validation experiments (E-C16 … E-C22),
the independent-seed validation arms (31–60), the anytime best-so-far study
(E-B2), the activated-health robustness sweeps (E-ALPHA / E-WROBUST), the
h0-coupled degradation replay (E-h0λ) and the hyper-parameter sensitivity
re-verification data (E1-TUNE), are included in the `results/` directory of
this repository (release v1.0.2). The seed manifest for the 1–60 protocol is
`results/seeds.json`. Raw simulation outputs beyond these files are available
from the corresponding author upon reasonable request.

## License

MIT — see [LICENSE](LICENSE).
