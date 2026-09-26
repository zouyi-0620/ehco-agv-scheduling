"""M20: FLC tie-break intervention rate on AW-NSGA-II (E1 protocol, 30 seeds).

Reruns AW-NSGA-II under the exact E1 scenario construction (same make_scenario
and the 30-seed main protocol, constants.SEEDS) and reports how often the FLC
scalarization decides a rank-tied binary tournament. Pure measurement; does not
modify e1 outputs.
"""
from __future__ import annotations

from .. import constants as C
from ..algorithms import run_algorithm
from ..scenario import make_scenario


def main() -> None:
    # Default = the 30-seed main protocol (constants.SEEDS).  seeds.json is a
    # 1-60 protocol manifest and deliberately does not drive runs (a2, 2026-09-24).
    seeds = list(C.SEEDS)
    rows = []
    for s in seeds:
        sc = make_scenario(s)
        res = run_algorithm("AW-NSGA-II", sc, s)
        p = res.params
        t_tot = int(p.get("flc_tournaments", 0))
        t_tie = int(p.get("flc_tiebreaks", 0))
        rate = t_tie / t_tot if t_tot else float("nan")
        rows.append((s, t_tot, t_tie, rate))
        print(f"seed {s:3d} tournaments={t_tot:8d} tiebreaks={t_tie:8d} "
              f"intervention={100*rate:5.2f}%", flush=True)
    mean_rate = sum(r[3] for r in rows) / len(rows)
    t_sum = sum(r[1] for r in rows)
    t_ties = sum(r[2] for r in rows)
    print(f"\n[M20] mean per-run tournaments={t_sum/len(rows):.0f}, "
          f"mean per-run tiebreaks={t_ties/len(rows):.0f}, "
          f"overall FLC intervention rate = {100*t_ties/t_sum:.2f}% "
          f"(seed-mean = {100*mean_rate:.2f}%)")


if __name__ == "__main__":
    main()
