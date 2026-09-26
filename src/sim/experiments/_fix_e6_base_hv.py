"""One-off fix: recompute the E6 baseline-row HV with the own-front basis.

The first E6 run reused the E1 union-basis HV for the `base` row while the
perturbation rows used the within-config own-front basis, which made every
% change vs base look like +22% (an artefact of the basis mismatch).

This script reloads the fixed `_load_base`, patches the `base` rows of
e6_per_run.csv, and re-aggregates e6_sensitivity.csv with the same logic as
run_e6().  Run from the project root.
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

from .. import constants as C
from .e6_sensitivity import (BASE, ORDER, RESULTS, PERTURBATIONS,
                             _load_base, _write_csv)


def _agg(per_run: list[dict]) -> list[dict]:
    agg = defaultdict(list)
    for r in per_run:
        agg[r["config"]].append(r)
    rows = []
    for name in ORDER:
        vals = agg[name]
        for k in ("f1", "f2", "f3", "f4", "hv", "delta_h_mean", "delta_h_max"):
            x = np.array([float(v[k]) for v in vals if float(v[k]) == float(v[k])],
                         dtype=float)
            if len(x) == 0:
                continue
            lo, hi = __import__("sim.metrics", fromlist=["ci95"]).ci95(x)
            b = np.array([float(v[k]) for v in agg[BASE] if float(v[k]) == float(v[k])],
                         dtype=float)
            if len(b) and b.mean() != 0:
                pct = 100.0 * (x.mean() - b.mean()) / abs(b.mean())
            else:
                pct = float("nan")
            rows.append({"config": name, "metric": k,
                         "mean": float(x.mean()), "std": float(x.std(ddof=1)),
                         "ci95_lo": lo, "ci95_hi": hi,
                         "pct_change_vs_base": round(pct, 2)})
    return rows


def main() -> None:
    seeds = list(C.SEEDS)
    base = _load_base(seeds)

    # patch e6_per_run.csv base rows
    per_path = os.path.join(RESULTS, "e6_per_run.csv")
    with open(per_path, encoding="utf-8") as f:
        cols = f.readline().strip().split(",")
        lines = [line.strip().split(",") for line in f]
    patched = 0
    for ln in lines:
        r = dict(zip(cols, ln))
        if r["config"] == BASE:
            s = int(r["seed"])
            r["hv"] = str(base[s]["hv"])
            ln[:] = [r[c] for c in cols]
            patched += 1
    assert patched == len(seeds), f"expected {len(seeds)} base rows, got {patched}"
    with open(per_path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for ln in lines:
            f.write(",".join(ln) + "\n")
    print(f"patched {patched} base rows in {per_path}")

    # re-aggregate
    rows = []
    with open(per_path, encoding="utf-8") as f:
        rd = __import__("csv").DictReader(f)
        rows = list(rd)
    agg = _agg(rows)
    _write_csv(os.path.join(RESULTS, "e6_sensitivity.csv"), agg)
    print("done")


if __name__ == "__main__":
    main()
