"""Fuzzy logic controller (SPEC.md section 7).

Inputs: rho (congestion), gamma (urgent-task ratio), h_bar (mean cumulative
health). Output: weights (w1..w4) summing to 1. Membership functions are the
exact trapezoid/triangle sets from the manuscript; the 27-rule consequent table
is the explicit, reproducible table documented in SPEC.md (review fix R1-M5).
"""
from __future__ import annotations

import numpy as np

from . import constants as C

# Global boundary scale for the membership functions (E6 sensitivity study:
# FLC boundary +/-10% perturbation). 1.0 = manuscript boundaries. Not part of
# the optimisation state; mutated only by experiment scripts.
BOUND_SCALE = 1.0


def _trap(x: float, a: float, b: float, c: float, d: float) -> float:
    s = BOUND_SCALE
    a, b, c, d = a * s, b * s, c * s, d * s
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    return (d - x) / (d - c) if d > c else 1.0


def _tri(x: float, a: float, b: float, c: float) -> float:
    s = BOUND_SCALE
    a, b, c = a * s, b * s, c * s
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    return (x - a) / (b - a) if x < b else (c - x) / (c - b)


def mu_rho(x: float) -> tuple[float, float, float]:
    return (_trap(x, 0, 0, 0.3, 0.5), _tri(x, 0.3, 0.5, 0.7),
            _trap(x, 0.5, 0.7, 1, 1))


def mu_gamma(x: float) -> tuple[float, float, float]:
    return (_trap(x, 0, 0, 0.2, 0.4), _tri(x, 0.2, 0.4, 0.6),
            _trap(x, 0.4, 0.6, 1, 1))


def mu_h(x: float) -> tuple[float, float, float]:
    return (_trap(x, 0, 0, 0.3, 0.5), _tri(x, 0.3, 0.5, 0.7),
            _trap(x, 0.5, 0.7, 1, 1))


# COG centroids of the output linguistic values L/M/H (manuscript P220)
CENTROID = np.array([0.067, 0.30, 0.50])

# 27-rule consequent table (SPEC.md section 7; explicit for reproducibility).
# index: (h_level, rho_level, gamma_level) -> (w1, w2, w3, w4)
# levels: 0=low/poor, 1=medium, 2=high/good
_RULES: dict[tuple[int, int, int], tuple[float, float, float, float]] = {}


def _build_rules() -> None:
    base = {
        # h = good (healthy fleet): balanced; efficiency priority under stress
        (2, 0, 0): (0.30, 0.25, 0.20, 0.25),
        (2, 2, 0): (0.50, 0.20, 0.15, 0.15),
        (2, 0, 2): (0.45, 0.25, 0.15, 0.15),
        (2, 2, 2): (0.60, 0.15, 0.10, 0.15),
        (2, 1, 0): (0.35, 0.25, 0.15, 0.25),
        (2, 0, 1): (0.35, 0.25, 0.15, 0.25),
        (2, 1, 1): (0.40, 0.20, 0.15, 0.25),
        (2, 1, 2): (0.50, 0.20, 0.10, 0.20),
        (2, 2, 1): (0.55, 0.15, 0.10, 0.20),
        # h = medium: health weight raised
        (1, 0, 0): (0.25, 0.20, 0.15, 0.40),
        (1, 2, 0): (0.40, 0.15, 0.10, 0.35),
        (1, 0, 2): (0.35, 0.20, 0.10, 0.35),
        (1, 2, 2): (0.45, 0.15, 0.05, 0.35),
        (1, 1, 0): (0.30, 0.20, 0.10, 0.40),
        (1, 0, 1): (0.30, 0.20, 0.10, 0.40),
        (1, 1, 1): (0.30, 0.20, 0.10, 0.40),
        (1, 1, 2): (0.40, 0.15, 0.10, 0.35),
        (1, 2, 1): (0.40, 0.15, 0.10, 0.35),
        # h = poor: equipment protection dominates
        (0, 0, 0): (0.15, 0.15, 0.10, 0.60),
        (0, 2, 0): (0.30, 0.10, 0.10, 0.50),
        (0, 0, 2): (0.25, 0.15, 0.10, 0.50),
        (0, 2, 2): (0.35, 0.10, 0.10, 0.45),
        (0, 1, 0): (0.20, 0.15, 0.10, 0.55),
        (0, 0, 1): (0.20, 0.15, 0.10, 0.55),
        (0, 1, 1): (0.20, 0.15, 0.10, 0.55),
        (0, 1, 2): (0.30, 0.15, 0.05, 0.50),
        (0, 2, 1): (0.30, 0.10, 0.10, 0.50),
    }
    _RULES.update(base)


_build_rules()


def flc_weights(rho: float, gamma: float, h_bar: float) -> np.ndarray:
    """Return normalised (w1, w2, w3, w4) with sum == 1 (COG + sum-normalise)."""
    mr, mg, mh = mu_rho(rho), mu_gamma(gamma), mu_h(h_bar)
    wsum = np.zeros(4)
    total = 0.0
    for hi, hm in enumerate(mh):
        if hm <= 0:
            continue
        for ri, rm in enumerate(mr):
            if rm <= 0:
                continue
            for gi, gm in enumerate(mg):
                if gm <= 0:
                    continue
                strength = min(hm, rm, gm)          # Mamdani AND (min)
                wsum += strength * np.array(_RULES[(hi, ri, gi)])
                total += strength
    if total <= 0 or wsum.sum() <= 0:
        return np.array([0.25, 0.25, 0.25, 0.25])
    return wsum / wsum.sum()


def scenario_features(assignments, tasks, h_cum) -> tuple[float, float, float]:
    """Compute (rho, gamma, h_bar) from current state.

    rho: fleet-activity ratio min(1, n_active/20) in [0,1] (NOT a spatial
    congestion measure; spatial congestion is handled by the aisle-occupancy
    trigger and the congestion-aware cost inflation, manuscript §2.5/§3.11);
    gamma: urgent-task ratio among pending tasks; h_bar: mean cumulative health.
    """
    h_bar = float(np.mean(h_cum)) if len(h_cum) else 1.0
    gamma = float(np.mean([t.omega > 0 for t in tasks])) if tasks else 0.0
    # fleet-activity proxy (manuscript §2.4.2); 0.5 for the nominal 10-AGV fleet
    rho = min(1.0, len(h_cum) / 20.0) if len(h_cum) else 0.0
    return rho, gamma, h_bar
