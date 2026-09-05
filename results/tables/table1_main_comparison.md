# Main Comparison (E1, 8 algorithms × 30 seeds) — manuscript Table 3

Archive regenerated 2026-09-04 from `sim/results/e1/e1_metrics.csv` (current M7-slack/W5 calibre). Values: mean ± SD (sample). HV-E1 = union-normalized hypervolume (protocol of manuscript §3.2/P171).

| Algorithm | f1 (s) | f2 (CNY) | f3 (kJ) | Δh̄ | Gini(Δh) | HV-E1 | vs AW-NSGA-II |
|---|---|---|---|---|---|---|---|
| A*-GA | 299.3 ± 11.0 | 2408.3 ± 138.0 | 1380.5 ± 58.6 | 0.093 ± 0.024 | 0.372 ± 0.071 | 0.120 ± 0.088 | *** |
| NSGA-II-2obj | 312.7 ± 24.0 | 2357.1 ± 142.8 | 1366.7 ± 59.7 | 0.091 ± 0.024 | 0.374 ± 0.077 | 0.222 ± 0.070 | *** |
| NSGA-II-3obj | 325.1 ± 33.8 | 2389.2 ± 174.9 | 1315.1 ± 64.4 | 0.089 ± 0.024 | 0.369 ± 0.068 | 0.550 ± 0.133 | ns |
| GA-SA | 297.9 ± 21.3 | 2357.9 ± 155.8 | 1288.8 ± 61.5 | 0.089 ± 0.024 | 0.372 ± 0.069 | 0.459 ± 0.193 | ns |
| NSGA-III | 330.8 ± 28.9 | 2433.4 ± 192.5 | 1322.5 ± 61.4 | 0.089 ± 0.026 | 0.367 ± 0.075 | 0.485 ± 0.140 | ns |
| MOEA/D | 285.0 ± 12.9 | 2220.2 ± 124.1 | 1307.2 ± 58.2 | 0.090 ± 0.024 | 0.370 ± 0.074 | 0.790 ± 0.145 | *** |
| LWC | 319.7 ± 34.5 | 2382.5 ± 141.2 | 1319.3 ± 71.8 | 0.090 ± 0.024 | 0.358 ± 0.067 | 0.533 ± 0.138 | ns |
| AW-NSGA-II | 326.8 ± 40.3 | 2449.0 ± 183.5 | 1320.8 ± 60.7 | 0.089 ± 0.024 | 0.377 ± 0.066 | 0.463 ± 0.143 | — |

*vs AW-NSGA-II = two-sided paired Wilcoxon on per-seed HV-E1, Holm–Bonferroni corrected over the 28-pair family (manuscript Table S7): *** p<sub>Holm</sub> < 0.001, ns otherwise. Bit-level disclosure: A*-GA and GA-SA return single-point fronts; their HV values carry a normalization convention (see manuscript).*
