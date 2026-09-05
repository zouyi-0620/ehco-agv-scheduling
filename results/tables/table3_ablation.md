# Ablation Study (E2, 7 variants × 30 seeds) — manuscript Table 5

Archive regenerated 2026-09-04 from `sim/results/e2/e2_metrics.csv` (current calibre). Values: mean ± SD (sample). HV-E2 = union-normalized over the seven variants' per-seed union (manuscript §3.4).

| Variant | f1 (s) | f2 (CNY) | Δh̄ | Gini(Δh) | HV-E2 | vs Full |
|---|---|---|---|---|---|---|
| Full | 326.8 ± 40.3 | 2449.0 ± 183.5 | 0.089 ± 0.024 | 0.377 ± 0.066 | 0.282 ± 0.097 | — |
| -health | 322.3 ± 36.9 | 2152.0 ± 179.2 | 0.089 ± 0.025 | 0.364 ± 0.078 | 0.611 ± 0.166 | *** |
| -adaptW | 327.5 ± 37.6 | 2410.3 ± 174.6 | 0.089 ± 0.024 | 0.362 ± 0.068 | 0.312 ± 0.119 | ns |
| -impAstar | 326.8 ± 40.3 | 2449.0 ± 183.5 | 0.089 ± 0.024 | 0.377 ± 0.066 | 0.282 ± 0.097 | ≡ Full (bit-identical) |
| -closedLoop | 326.8 ± 40.3 | 2449.0 ± 183.5 | 0.089 ± 0.024 | 0.377 ± 0.066 | 0.282 ± 0.097 | ≡ Full (bit-identical) |
| hard-only | 322.3 ± 36.9 | 2152.0 ± 179.2 | 0.089 ± 0.025 | 0.364 ± 0.078 | 0.611 ± 0.166 | *** |
| GA-SA-multiSoH | 297.9 ± 21.3 | 2357.9 ± 155.8 | 0.089 ± 0.024 | 0.372 ± 0.069 | 0.317 ± 0.150 | ns |

*vs Full = two-sided paired Wilcoxon on per-seed HV-E2, Holm–Bonferroni over the 4 non-identical comparisons: *** p<sub>Holm</sub> < 0.001, ns otherwise. Static-scenario equivalences (bit-level, §3.4): −health ≡ hard-only; −impA* ≡ −closedLoop ≡ Full (HV-E2 differences zero; Wilcoxon undefined).*
