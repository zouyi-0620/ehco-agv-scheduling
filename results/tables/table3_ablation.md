# Table 3: Ablation Study (E2, 30 seeds, mean ± SD)

| Variant | f1 (s) | f2 (CNY) | Δh_mean | Gini(Δh) | HV | vs Full |
|---------|--------|----------|---------|----------|----|---------|
| Full | 335.0 ± 45.8 | 2256.0 ± 196.3 | 0.1 ± 0.0 | 0.4 ± 0.1 | 0.3 ± 0.1 | — |
| -health | 327.3 ± 34.1 | 1947.1 ± 171.6 | 0.1 ± 0.0 | 0.4 ± 0.1 | 0.6 ± 0.1 | ns |
| -adaptW | 329.8 ± 41.2 | 2274.0 ± 160.4 | 0.1 ± 0.0 | 0.4 ± 0.1 | 0.3 ± 0.1 | ns |
| -impAstar | 335.0 ± 45.8 | 2256.0 ± 196.3 | 0.1 ± 0.0 | 0.4 ± 0.1 | 0.3 ± 0.1 | ns |
| -closedLoop | 335.0 ± 45.8 | 2256.0 ± 196.3 | 0.1 ± 0.0 | 0.4 ± 0.1 | 0.3 ± 0.1 | ns |
| hard-only | 327.3 ± 34.1 | 1947.1 ± 171.6 | 0.1 ± 0.0 | 0.4 ± 0.1 | 0.6 ± 0.1 | ns |
| GA-SA-multiSoH | 289.9 ± 13.9 | 2176.5 ± 155.6 | 0.1 ± 0.0 | 0.4 ± 0.1 | 0.4 ± 0.2 | ns |

*HV normalised over the union of all 7 variant fronts (E2). Static-scenario equivalences: −health ≡ hard-only; −impA* ≡ −closedLoop ≡ Full.*