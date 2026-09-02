"""E6 parameter-sensitivity study (SPEC.md section 9, E6; Table S2 / Fig 7).

AW-NSGA-II is re-run under one-at-a-time parameter perturbations; every
configuration shares the same 30 fixed seeds and the equal 20,000-eval budget.

Perturbation grid (SPEC E6):
  h_safe        +/-10%           (0.54 / 0.66)
  alpha         {0.3, 0.4}       (baseline 0.5; SPEC range [0.3, 0.5])
  beta (EWMA)   {5e-4, 5e-3}     (baseline 1e-3)
  AHP weights   x0.8 / x1.2      (renormalised to sum 1)
  FLC bounds    +/-10%           (membership boundaries scaled)
  degradation threshold +/-30%   (-7e-5 / -1.3e-4)
  deadline mult {1.3, 1.7}       (baseline 1.5)

Honest reporting notes (static scenario):
  - h_safe / alpha / degradation threshold affect only the piecewise health
    penalty f4 and the warning rule; with h_cum in [0.7, 1.0] the static f4 is
    0 for every perturbation, so HV/f1 are structurally insensitive (reported
    as such; the dynamic scenarios E4 exercise these components).
  - beta / AHP weights affect the EWMA health estimator -> the fleet delta-h
    statistics, not the static fitness itself (reported via delta_h metrics).
  - FLC bounds and deadline multiplier change the optimisation landscape ->
    HV/f1 responses.

Baseline row is reused from E1 (AW-NSGA-II) so Table S2 aligns with Table 1.

Outputs (results/e6/):
  e6_sensitivity.csv : per config (mean, std, 95% CI) of hv, f1..f4,
                       delta_h_mean, delta_h_max + % change vs baseline
  e6_per_run.csv     : raw per (config, seed) rows for downstream figures

Run from the project root:  python -m sim.experiments.e6_sensitivity
"""
from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict

import numpy as np

from .. import constants as C
from .. import flc as FLC
from ..algorithms import aw_nsga2
from ..algorithms.common import EvalBudget, fast_non_dominated_sort, finite_F
from ..metrics import ci95, hypervolume
from ..objectives import EvalConfig
from ..scenario import make_scenario
from .e1_main import RESULTS as E1_DIR, _fleet_health, _hv_in_subspace

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "e6")

BASE = "base"

# config name -> (dict of C attribute overrides, FLC bound scale)
PERTURBATIONS: dict[str, tuple[dict, float]] = {
    "h_safe_lo":  ({"H_SAFE": 0.54}, 1.0),
    "h_safe_hi":  ({"H_SAFE": 0.66}, 1.0),
    "alpha_lo":   ({"ALPHA_PENALTY": 0.3}, 1.0),
    "alpha_mid":  ({"ALPHA_PENALTY": 0.4}, 1.0),
    "beta_lo":    ({"EWMA_BETA": 5e-4}, 1.0),
    "beta_hi":    ({"EWMA_BETA": 5e-3}, 1.0),
    "ahp_lo":     ({"AHP_W": np.array([0.32, 0.28, 0.20])}, 1.0),   # x0.8, renorm
    "ahp_hi":     ({"AHP_W": np.array([0.48, 0.42, 0.30])}, 1.0),   # x1.2, renorm
    "flc_lo":     ({}, 0.9),
    "flc_hi":     ({}, 1.1),
    "thr_lo":     ({"DEGRAD_THRESHOLD": -7e-5}, 1.0),
    "thr_hi":     ({"DEGRAD_THRESHOLD": -1.3e-4}, 1.0),
    "dl_lo":      ({"DEADLINE_MULT": 1.3}, 1.0),
    "dl_hi":      ({"DEADLINE_MULT": 1.7}, 1.0),
}
ORDER = [BASE] + list(PERTURBATIONS)

_ORIG: dict = {}


def _apply(name: str) -> None:
    """Apply a perturbation; store originals for _restore."""
    global _ORIG
    if not _ORIG:
        _ORIG = {"C": {k: getattr(C, k) for k in
                       ("H_SAFE", "ALPHA_PENALTY", "EWMA_BETA", "AHP_W",
                        "DEGRAD_THRESHOLD", "DEADLINE_MULT")},
                 "FLC_BOUND_SCALE": FLC.BOUND_SCALE}
    ov, scale = PERTURBATIONS[name]
    for k, v in ov.items():
        setattr(C, k, v)
    FLC.BOUND_SCALE = scale


def _restore() -> None:
    global _ORIG
    for k, v in _ORIG["C"].items():
        setattr(C, k, v)
    FLC.BOUND_SCALE = _ORIG["FLC_BOUND_SCALE"]
    _ORIG = {}


def _load_base(seeds: list[int]) -> dict[int, dict]:
    """AW-NSGA-II metrics from E1 for the baseline row.

    NOTE (HV basis): the E1 e1_metrics.csv 'hv' column is the D5 union basis
    (min-max over the union of ALL 8 algorithms' fronts), which is NOT
    comparable with the within-config own-front basis used for the E6
    perturbation rows.  To keep the tornado plot's %-change honest, the
    baseline HV is recomputed here from the saved AW-NSGA-II fronts
    (e1_solutions.csv) using the SAME own-front normalisation as run_e6().
    """
    met_path = os.path.join(E1_DIR, "e1_metrics.csv")
    sol_path = os.path.join(E1_DIR, "e1_solutions.csv")
    if not os.path.exists(met_path) or not os.path.exists(sol_path):
        raise FileNotFoundError(
            "E1 results missing - run `python -m sim.experiments.e1_main` first")
    # per-seed non-HV metrics
    out: dict[int, dict] = {}
    with open(met_path, encoding="utf-8") as f:
        cols = f.readline().strip().split(",")
        for line in f:
            r = dict(zip(cols, line.strip().split(",")))
            if r["algo"] != "AW-NSGA-II":
                continue
            s = int(r["seed"])
            out[s] = {k: float(r[k]) for k in
                      ("f1", "f2", "f3", "f4", "delta_h_mean",
                       "delta_h_max", "gini_delta_h")}
    # per-seed front -> own-front normalised 4-D HV (same code path as run_e6)
    fronts: dict[int, list[list[float]]] = defaultdict(list)
    with open(sol_path, encoding="utf-8") as f:
        cols = f.readline().strip().split(",")
        for line in f:
            r = dict(zip(cols, line.strip().split(",")))
            if r["algo"] != "AW-NSGA-II":
                continue
            s = int(r["seed"])
            fronts[s].append([float(r[k]) for k in ("f1", "f2", "f3", "f4")])
    for s, pts in fronts.items():
        F = np.asarray(pts)
        lo, hi = F.min(axis=0), F.max(axis=0)
        span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        Fn = (F - lo) / span
        out[s]["hv"] = float(hypervolume(Fn, np.ones(4)))
    missing = [s for s in seeds if s not in out]
    if missing:
        raise FileNotFoundError(f"E1 AW-NSGA-II incomplete for seeds {missing}")
    return out


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)")


def run_e6(seeds: list[int] | None = None, verbose: bool = True) -> dict:
    seeds = seeds or list(C.SEEDS)
    base = _load_base(seeds)
    cfg = EvalConfig()
    os.makedirs(RESULTS, exist_ok=True)

    per_run: list[dict] = []
    t_start = time.perf_counter()
    for name in PERTURBATIONS:
        _apply(name)
        for s in seeds:
            sc = make_scenario(s)
            t0 = time.perf_counter()
            res = aw_nsga2.run(sc, s, cfg=cfg, budget=EvalBudget())
            idx = fast_non_dominated_sort(finite_F(res.F_final))[0]
            F = res.F_final[idx]
            # 4-D HV normalised by this run's own front (within-config basis;
            # relative changes vs base are what the tornado plot needs)
            lo, hi = F.min(axis=0), F.max(axis=0)
            span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
            Fn = (F - lo) / span
            hv = hypervolume(Fn, np.ones(4))
            fh = _fleet_health(res, sc)
            per_run.append({
                "config": name, "seed": s,
                "f1": float(res.best_F[0]), "f2": float(res.best_F[1]),
                "f3": float(res.best_F[2]), "f4": float(res.best_F[3]),
                "hv": float(hv),
                "delta_h_mean": fh["delta_h_mean"],
                "delta_h_max": fh["delta_h_max"],
                "wall_s": round(res.wall_s, 3),
            })
            if verbose:
                print(f"  {name:<10} seed {s:3d} "
                      f"{time.perf_counter()-t0:5.2f}s  "
                      f"hv={hv:.4f} f1={res.best_F[0]:7.1f}s", flush=True)
        _restore()
    if verbose:
        print(f"[E6] {len(PERTURBATIONS)} configs x {len(seeds)} seeds: "
              f"{time.perf_counter()-t_start:.1f}s", flush=True)

    # baseline rows from E1 (same seeds, same metrics)
    for s in seeds:
        per_run.append({"config": BASE, "seed": s,
                        "f1": base[s]["f1"], "f2": base[s]["f2"],
                        "f3": base[s]["f3"], "f4": base[s]["f4"],
                        "hv": base[s]["hv"],
                        "delta_h_mean": base[s]["delta_h_mean"],
                        "delta_h_max": base[s]["delta_h_max"],
                        "wall_s": float("nan")})
    _write_csv(os.path.join(RESULTS, "e6_per_run.csv"), per_run)

    # aggregation: mean +- std, 95% CI, % change vs base
    agg = defaultdict(list)
    for r in per_run:
        agg[r["config"]].append(r)
    rows = []
    for name in ORDER:
        vals = agg[name]
        for k in ("f1", "f2", "f3", "f4", "hv", "delta_h_mean", "delta_h_max"):
            x = np.array([v[k] for v in vals if np.isfinite(v[k])], dtype=float)
            if len(x) == 0:
                continue
            lo, hi = ci95(x)
            b = np.array([v[k] for v in agg[BASE] if np.isfinite(v[k])],
                         dtype=float)
            if len(b) and b.mean() != 0:
                pct = 100.0 * (x.mean() - b.mean()) / abs(b.mean())
            else:
                pct = float("nan")
            rows.append({"config": name, "metric": k,
                         "mean": float(x.mean()), "std": float(x.std(ddof=1)),
                         "ci95_lo": lo, "ci95_hi": hi,
                         "pct_change_vs_base": round(pct, 2)})
    _write_csv(os.path.join(RESULTS, "e6_sensitivity.csv"), rows)

    return {"n_configs": len(ORDER), "n_seeds": len(seeds),
            "out_dir": RESULTS, "rows": len(per_run)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="comma list, e.g. 1,2,3")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    run_e6(seeds=seeds)


if __name__ == "__main__":
    main()
