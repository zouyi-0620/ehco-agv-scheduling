# Parameter Sensitivity (E6, 14 perturbations × 30 seeds) — Table S2

Archive regenerated 2026-09-04 from `sim/results/e6/e6_sensitivity.csv` (current calibre); identical values to the rebuilt supplementary Table S2 and manuscript Table 2.

| Perturbation | f1 (%Δ) | f2 (%Δ) | HV (%Δ) | Δh̄ (%Δ) |
|---|---|---|---|---|
| base | +0.00 | +0.00 | +0.00 | +0.00 |
| h_safe_lo | +0.00 | +0.00 | +0.00 | +0.00 |
| h_safe_hi | +0.00 | +0.00 | +0.00 | +0.00 |
| alpha_lo | +0.00 | +0.00 | +0.00 | +0.00 |
| alpha_mid | +0.00 | +0.00 | +0.00 | +0.00 |
| beta_lo | +0.00 | +0.00 | +0.00 | -26.77 |
| beta_hi | +0.00 | +0.00 | +0.00 | +25.38 |
| ahp_lo | +0.00 | +0.00 | +0.00 | +169.84 |
| ahp_hi | +0.00 | +0.00 | +0.00 | -169.84 |
| flc_lo | +1.16 | -2.21 | +1.73 | -0.23 |
| flc_hi | -1.82 | -2.32 | -4.59 | +0.39 |
| thr_lo | +0.00 | +0.00 | +0.00 | +0.00 |
| thr_hi | +0.00 | +0.00 | +0.00 | +0.00 |
| dl_lo | -3.70 | +6.61 | +0.41 | +0.41 |
| dl_hi | -0.78 | -9.88 | -2.79 | +0.25 |

*Base = AW-NSGA-II default parameters (solution unchanged, Δf1 = ΔHV = 0.0%, for h_safe, α, β and AHP rows — β/AHP Δh̄ shifts reflect SoH-reading-only changes). FLC-boundary ±10%: Δf1 −1.8%/+1.2%, ΔHV −4.6%/+1.7%. Deadline multiplier most sensitive: Δf1 −3.7% (dl_lo).*
