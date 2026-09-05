# -*- coding: utf-8 -*-
"""E-B2: strict fixed-wall-clock best-so-far (anytime) study (expert-8, B2).

Question
--------
Under the real-time wall-clock budgets the online closed-loop layer demands
(100/200/500/1000 ms, cf. the 200 ms local / 1 s global design budgets of the
manuscript), how much solution quality can the EHCO online core (AW-NSGA-II)
deliver as a function of allowed wall-clock time, and how does that compare
with the MOEA/D core of the moead-cl baseline?

Protocol (E1-compatible, full comparability)
--------------------------------------------
* Every one of the EIGHT algorithms is run once per seed at its native
  20,000-evaluation budget (identical to E1).  AW-NSGA-II and MOEA/D
  additionally report every generation end: elapsed wall time + a copy of
  the current population objectives (tracker, default-off; behaviour-neutral,
  verified by a bit-exact knee-objective gate d_obj == 0 vs e1 metrics).
* The per-seed HV normalisation reference is the exact E1 D5 union: the
  min/max of the non-dominated fronts of all eight final populations on that
  seed, recomputed here.  Union bounds are seed-fixed, hence identical for
  both cores at every time point.
* Any time t, best-so-far HV = cumulative max of the hypervolume of the
  algorithm's current non-dominated front (same normalisation), evaluated on
  generation ends sampled on a 50 ms time grid (plus the final generation);
  every budget grid point is a multiple of 50 ms.
* The trajectory therefore terminates at (within ~5e-4 of) the E1 hv of
  Table 3; the residual is the population-level float tolerance of the six
  non-core algorithms' reruns used only to rebuild the union reference.

Sanity gates
------------
(1) reruns reproduce the E1/E1-indep knee objectives bit-exactly (d_obj == 0);
(2) final best-so-far HV matches e1_metrics 'hv' within 5e-4.

Outputs (sim/results/e_b2_anytime/)
-----------------------------------
  e_b2_trajectory.csv   seed, algo, t_ms, hv_now, bsf_hv   (50 ms grid)
  e_b2_budget_bsf.csv   seed, algo, budget_ms, bsf_hv
  e_b2_agg.csv / e_b2_stats.csv   mean+/-sd; paired Wilcoxon + Holm(5)
  e_b2_convergence.csv  per seed/algo: t to 50/90/99% of own final HV
  e_b2_crossover.csv    per seed: first t (<= 2000 ms) where MOEA/D bsf
                        strictly exceeds AW bsf
  e_b2_parity.csv       per seed/algo d_obj / d_hv_final gates
  e_b2_meta.json

Run:  python -m sim.experiments.e_b2_anytime --seeds "1..30"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict

import numpy as np

from .. import constants as C
from ..algorithms import ORDER, run_algorithm
from ..metrics import fast_non_dominated_sort, hypervolume
from ..scenario import make_scenario

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results",
                       "e_b2_anytime")
BUDGET_MS = (100, 200, 500, 1000, 2000)
ALGOS = ("AW-NSGA-II", "MOEA/D")
REF = np.ones(4)
LARGE = 1e9
SAMPLE_MS = 50.0            # HV sampling grid (ms)
GATE_OBJ = 1e-12
GATE_HV = 5e-4


def _nd_front(F: np.ndarray) -> np.ndarray:
    Ff = np.where(np.isinf(F), LARGE, F)
    fr = fast_non_dominated_sort(Ff)[0]
    keep = np.isfinite(F[fr]).all(axis=1)
    return Ff[fr][keep]


def _front_hv(F: np.ndarray, lo: np.ndarray, span: np.ndarray) -> float:
    fr = _nd_front(F)
    if len(fr) == 0:
        return 0.0
    return hypervolume((fr - lo) / span, REF)


def _wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    from scipy import stats
    return float(stats.wilcoxon(x, y, alternative="two-sided").pvalue)


def _holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    out = np.ones_like(p)
    acc = 1.0
    for rank in range(len(p) - 1, -1, -1):
        i = order[rank]
        acc = min(acc, p[i] * (rank + 1))
        out[i] = acc
    return out


def _parse_metrics_csv(path: str) -> dict:
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(int(r["seed"]), r["algo"])] = {
                k: float(r[k]) for k in ("f1", "f2", "f3", "f4", "hv")}
    return out


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)", flush=True)


def run_b2(seeds: list[int], verbose: bool = True) -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    base = os.path.join(os.path.dirname(RESULTS))
    # reference metrics: seeds 1-30 live in e1/, seeds 31-60 in e1_indep/
    ref = {}
    for sub in ("e1", "e1_indep"):
        ref.update(_parse_metrics_csv(os.path.join(base, sub,
                                                   "e1_metrics.csv")))

    traj_rows: list[dict] = []
    budget_rows: list[dict] = []
    conv_rows: list[dict] = []
    cross_rows: list[dict] = []
    parity_rows: list[dict] = []
    parity_fail = 0
    hv_fail = 0

    t_start = time.perf_counter()
    for s in seeds:
        sc = make_scenario(s)
        snaps: dict[str, list] = {}          # algo -> [(elapsed_s, F)] per gen
        finals: dict[str, np.ndarray] = {}
        for name in ORDER:
            def tracker(g, elapsed, F, _n=name):
                snaps.setdefault(_n, []).append((elapsed, F))
            kw = {"tracker": tracker} if name in ALGOS else {}
            res = run_algorithm(name, sc, s, **kw)
            finals[name] = res.F_final
            if name in ALGOS:
                exp = ref[(s, name)]
                d_obj = max(abs(res.best_F[k] - exp["f{}".format(k + 1)])
                            for k in range(4))
                if d_obj > GATE_OBJ:
                    parity_fail += 1
                parity_rows.append({"seed": s, "algo": name, "d_obj": d_obj})
        # exact E1 D5 union reference from the eight final populations
        union = np.vstack([_nd_front(finals[n]) for n in ORDER])
        lo = union.min(axis=0)
        hi = union.max(axis=0)
        span = np.where(hi - lo > 1e-12, hi - lo, 1.0)

        seed_traj: dict[str, list] = {a: [] for a in ALGOS}
        for name in ALGOS:
            tl = snaps[name]
            times = np.array([el * 1000.0 for (el, _F) in tl])
            max_t = max(2000.0, float(times[-1]))
            # sample generation ends on a SAMPLE_MS grid (incl. the last gen)
            marks = np.arange(SAMPLE_MS, max_t + SAMPLE_MS, SAMPLE_MS)
            keep = np.unique(np.searchsorted(times, marks, side="right") - 1)
            keep = keep[(keep >= 0) & (keep < len(tl))]
            if len(keep) == 0 or keep[-1] != len(tl) - 1:
                keep = np.append(keep, len(tl) - 1)
            hvs = [_front_hv(tl[int(k)][1], lo, span) for k in keep]
            t_keep = times[keep]
            bsf = np.maximum.accumulate(np.asarray(hvs, dtype=float))
            seed_traj[name] = [t_keep, bsf]
            for (tk, hk, bk) in zip(t_keep, hvs, bsf):
                traj_rows.append({"seed": s, "algo": name,
                                  "t_ms": round(float(tk), 3),
                                  "hv_now": float(hk), "bsf_hv": float(bk)})
            # gates
            d_hv = float(bsf[-1]) - ref[(s, name)]["hv"]
            for pr in parity_rows:
                if pr["seed"] == s and pr["algo"] == name:
                    pr["d_hv_final"] = d_hv
            if abs(d_hv) > GATE_HV:
                hv_fail += 1
            final = float(bsf[-1])
            for b in BUDGET_MS:
                k = int(np.searchsorted(t_keep, b, side="right")) - 1
                budget_rows.append({"seed": s, "algo": name, "budget_ms": b,
                                    "bsf_hv": float(bsf[k]) if k >= 0 else 0.0})
            for frac, lab in ((0.5, "t50"), (0.9, "t90"), (0.99, "t99")):
                hit = np.where(bsf >= final * frac)[0]
                conv_rows.append({"seed": s, "algo": name, "metric": lab,
                                  "t_ms": float(t_keep[hit[0]]) if len(hit)
                                  else float("nan"),
                                  "final_hv": final})
        # crossover within 2000 ms on the joint 50 ms grid
        def curve(a):
            return seed_traj[a][0], seed_traj[a][1]
        ta, ha = curve("AW-NSGA-II")
        tm, hm = curve("MOEA/D")
        tj = np.arange(0.0, 2000.0 + SAMPLE_MS, SAMPLE_MS)
        ha_i = np.interp(tj, ta, ha, left=0.0, right=ha[-1])
        hm_i = np.interp(tj, tm, hm, left=0.0, right=hm[-1])
        ahead = hm_i > ha_i + 1e-12
        t_cross = float(tj[np.argmax(ahead)]) if ahead.any() else float("inf")
        cross_rows.append({"seed": s, "t_cross_ms": t_cross,
                           "aw_final": float(ha[-1]),
                           "moead_final": float(hm[-1]),
                           "moead_exceeds_aw_2s": int(ahead.any())})
        if verbose and (s == seeds[0] or s == seeds[-1] or len(seeds) <= 4):
            print(f"  seed {s:3d} cross={t_cross:7.1f}ms "
                  f"awF={float(ha[-1]):.4f} moeF={float(hm[-1]):.4f} "
                  f"({time.perf_counter()-t_start:.0f}s)", flush=True)

    _write_csv(os.path.join(RESULTS, "e_b2_trajectory.csv"), traj_rows)
    _write_csv(os.path.join(RESULTS, "e_b2_budget_bsf.csv"), budget_rows)
    _write_csv(os.path.join(RESULTS, "e_b2_convergence.csv"), conv_rows)
    _write_csv(os.path.join(RESULTS, "e_b2_crossover.csv"), cross_rows)
    for pr in parity_rows:
        pr.setdefault("d_hv_final", float("nan"))
    _write_csv(os.path.join(RESULTS, "e_b2_parity.csv"), parity_rows)

    agg = defaultdict(lambda: defaultdict(list))
    for r in budget_rows:
        agg[r["algo"]][r["budget_ms"]].append(r["bsf_hv"])
    agg_rows, stat_rows = [], []
    for b in BUDGET_MS:
        a = np.asarray(agg["AW-NSGA-II"][b], dtype=float)
        m = np.asarray(agg["MOEA/D"][b], dtype=float)
        agg_rows.append({"algo": "AW-NSGA-II", "budget_ms": b,
                         "mean": float(a.mean()), "std": float(a.std(ddof=1)),
                         "n": len(a)})
        agg_rows.append({"algo": "MOEA/D", "budget_ms": b,
                         "mean": float(m.mean()), "std": float(m.std(ddof=1)),
                         "n": len(m)})
        stat_rows.append({"budget_ms": b, "mean_aw": float(a.mean()),
                          "mean_moead": float(m.mean()),
                          "n": len(a),
                          "wilcoxon_p": _wilcoxon(a, m)})
    pv = np.asarray([r["wilcoxon_p"] for r in stat_rows])
    for r, ph in zip(stat_rows, _holm(pv)):
        r["holm_p_5"] = ph
    _write_csv(os.path.join(RESULTS, "e_b2_agg.csv"), agg_rows)
    _write_csv(os.path.join(RESULTS, "e_b2_stats.csv"), stat_rows)

    meta = {
        "protocol": ("8 algorithms run once per seed at their native 20k-eval "
                     "budget (identical to E1); AW-NSGA-II & MOEA/D log every "
                     "generation end (elapsed wall time + population copy); "
                     "best-so-far HV sampled on a 50 ms generation-end grid "
                     "with the exact E1 D5 union reference rebuilt from the "
                     "eight final populations; trajectory ends within 5e-4 of "
                     "the Table-3 hv"),
        "sample_ms": SAMPLE_MS, "budgets_ms": list(BUDGET_MS),
        "algorithms": list(ALGOS), "seeds": seeds,
        "reference_metrics": "e1_metrics.csv (e1: seeds 1-30; "
                             "e1_indep: seeds 31-60)",
        "gates": {"d_obj": GATE_OBJ, "d_hv_final": GATE_HV},
        "gates_hit": {"parity_fail": parity_fail, "hv_fail": hv_fail},
        "scipy": "1.18.1",
    }
    with open(os.path.join(RESULTS, "e_b2_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=1, default=str)
    print(f"[E-B2] {len(seeds)} seeds x {len(ORDER)} algos "
          f"{time.perf_counter()-t_start:.0f}s -> {RESULTS} "
          f"(parity_fail={parity_fail} hv_fail={hv_fail})", flush=True)
    return {"n_seeds": len(seeds), "out_dir": RESULTS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None,
                    help="comma list, e.g. 1,2,3; default 1..60")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds \
        else list(range(1, 61))
    run_b2(seeds=seeds)


if __name__ == "__main__":
    main()
