# e1_tune — baseline hyper-parameter sensitivity (E1 static protocol)

Data backing the baseline-hyper-parameter sensitivity statements in the
manuscript (§2.4.4, §3.1, §3.2):

| file | config | 30-seed mean HV-E1 | default (e1/e1_metrics.csv) |
|---|---|---|---|
| metrics_MOEA_D-T10.csv | MOEA/D neighbourhood T = 10 | 0.671 | 0.790 (T = 20) |
| metrics_MOEA_D-T30.csv | MOEA/D neighbourhood T = 30 | 0.834 | 0.790 (T = 20) |
| metrics_NSGA-III-H4.csv | NSGA-III Das–Dennis H = 4 | 0.478 | 0.485 (H = 5) |
| metrics_NSGA-III-H6.csv | NSGA-III Das–Dennis H = 6 | 0.482 | 0.485 (H = 5) |

Protocol: identical to E1 (20,000 evaluations/run, population 100, seeds
1–30); HV is the per-seed union-normalised HV-E1 of the manuscript (bounds
rebuilt from the archived eight-algorithm final populations in
`e1/e1_solutions.csv`). Regenerated 2026-09-05 from the current engine
(`python _exp12_hyper.py`; raw rows also in `e1_tune_verify/hyper_verify.csv`).

Historical note: an earlier revision of these files (backed up in
`backup_pre_20260905/`) was produced under an older scenario revision and is
NOT comparable with the current manuscript numbers — do not use it.
