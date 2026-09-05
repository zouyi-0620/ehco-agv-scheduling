# Task Load Distribution by SoH Group (E-fig4, 30 seeds) — Table S4

Archive regenerated 2026-09-04 from `sim/results/e_fig4/{taskload_summary.csv, spearman.csv}` (current calibre, route B). Per-seed Spearman ρ: one-sample t-test across 30 seeds, Bonferroni-corrected for the three-configuration family.

| Variant | Low SoH [0.70–0.80) | Mid SoH [0.80–0.90) | High SoH [0.90–1.00] | Spearman ρ | raw p | Bonf p |
|---|---|---|---|---|---|---|
| Full EHCO | 4.97 ± 0.59 (n=29) | 4.97 ± 0.49 (n=30) | 5.25 ± 0.60 (n=30) | +0.123 ± 0.307 | 0.028 | 0.0839 |
| − Equipment health constraint (= hard-only, bit-identical) | 5.17 ± 0.57 (n=29) | 5.04 ± 0.38 (n=30) | 4.86 ± 0.51 (n=30) | -0.129 ± 0.301 | 0.0193 | 0.0578 |
| GA-SA (multi-indicator SoH, hard exclusion) | 5.05 ± 0.54 (n=29) | 5.01 ± 0.49 (n=30) | 4.98 ± 0.43 (n=30) | +0.053 ± 0.368 | 0.431 | 1 |

*n = 30 independent seeds (Low-SoH column may carry n < 30 when one seed had a degenerate uniform allocation; full per-seed data are in `taskload_per_run.csv`). Raw one-sample t p values and Bonferroni-adjusted p values for the three-configuration family are reported. After correction, none remains significant (|ρ| < 0.15 with opposite signs for the two AW-NSGA-II arms), consistent with f4 ≡ 0 in the static scenario.*
