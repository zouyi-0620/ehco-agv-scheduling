# Table S6: Objective Correlation & PCA (AW-NSGA-II final front, pooled n = 1,057)

Archive checked 2026-09-11 — values match the current calibre (manuscript §2.5 P64/P66, supplementary Table S6 and Figure S1; Spearman ρ = 0.327/0.507/0.615; PCA λ = 1.975/0.682/0.342 computed on the *same* f1–f3 Spearman correlation matrix shown below, Kaiser effective dimensionality 1). Earlier calibres archived under `backup_pre_20260904/`.

## Spearman ρ matrix (pooled over the 30-run AW-NSGA-II final fronts)
|     | f1 | f2 | f3 | f4 |
|-----|----|----|----|----|
| f1 | 1.000 | 0.327 | 0.507 | N/A |
| f2 | 0.327 | 1.000 | 0.615 | N/A |
| f3 | 0.507 | 0.615 | 1.000 | N/A |
| f4 | N/A | N/A | N/A | 1.000 |

## PCA eigenvalues (f1–f3 Spearman correlation matrix — same matrix as the ρ table above)
| PC | Eigenvalue | Explained Var (%) |
|----|------------|-------------------|
| PC1 | 1.975 | 65.8% |
| PC2 | 0.682 | 22.7% |
| PC3 | 0.342 | 11.4% |

*Effective dimensionality (Kaiser > 1): 1. f4 is structurally constant (0) in the static scenario. PCA is taken on the Spearman matrix so that the eigenvalue block and the ρ block above share one estimator; the earlier archive reported λ on the Pearson matrix (1.907/0.694/0.400, 63.6%) and has been superseded.*
