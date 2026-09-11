# E-B2 anytime best-so-far study — file conventions

This directory stores the strict fixed-wall-clock best-so-far (E-B2) outputs
that support Table S12 and Figure S3 of the supplementary material.

## Which file to use for the reported p-values (important)

Table S12 / Figure S3 statistics are computed from `e_b2_trajectory.csv` using the
*generation-end* `bsf_hv` sample whose wall-clock time is the last generation end
`<= budget` (`np.searchsorted(t, budget, side="right") - 1`).  **The per-arm series
must be built over the deduplicated seed list** (`sorted({s for (s, algo) in traj if s <= 30})`,
30 seeds per arm) — `traj` is keyed by `(seed, algo)`, so naively iterating its keys
duplicates every seed and inflates the paired Wilcoxon statistic.  The table builds
with a standard Holm–Bonferroni step-down (smallest p multiplied by m = 5).
See `src/sim/experiments/e_b2_anytime.py` and the table builder (`_add_b2_suppl.py`).

Note (2026-09-10): an earlier revision of this note claimed that `e_b2_budget_bsf.csv`
(exact 50 ms grid) yields different p-values and must not be used.  That is **obsolete**:
the two files currently agree exactly at every budget point (0/600 mismatches), so either
gives the same per-seed values; only the deduplication and Holm conventions above matter.

## Files

| File | Content | Used by |
|---|---|---|
| `e_b2_trajectory.csv` | per-seed, per-generation-end elapsed wall time + current-front HV (`hv_now`) and cumulative-max best-so-far HV (`bsf_hv`) | Table S12 / Figure S3 statistics (searchsorted lookup) |
| `e_b2_budget_bsf.csv` | per-seed `bsf_hv` sampled on the exact 50 ms budget grid (100/200/500/1000/2000 ms) | convenience export / figure curve points |
| `e_b2_agg.csv` | mean ± SD of `bsf_hv` per algorithm/budget over all 60 seeds | aggregate overview |
| `e_b2_stats.csv` | per-budget Wilcoxon (two-sided paired) and Holm(5) over the pooled 60 seeds | protocol summary (not the per-arm Table S12 rows) |
| `e_b2_convergence.csv` | per-seed times to reach 50/90/99% of the final HV | convergence analysis |
| `e_b2_crossover.csv` | per-seed wall-clock time when MOEA/D best-so-far exceeds AW-NSGA-II | crossover analysis (§3.8) |
| `e_b2_parity.csv` | per-seed parity check against Table 3 HV values | reproducibility audit |
| `e_b2_meta.json` | protocol description, seed lists, budgets | provenance |
