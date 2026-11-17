| Method | Alive | Eligible | Purity (%) | Purity minus shuffle (points) | Reconstruction EV (%) | Positive units per slice |
| --- | --- | --- | --- | --- | --- | --- |
| Supervised | 424.1 ± 86.5 | 374.9 ± 83.6 | 79.82 ± 0.46 | 23.93 ± 0.47 | 99.862 ± 0.061 | 24.00 ± 0.00 |
| SimCLR | 1309.3 ± 87.0 | 1238.1 ± 77.5 | 79.87 ± 1.06 | 24.01 ± 1.09 | 99.687 ± 0.032 | 23.98 ± 0.00 |
| NCL | 198.0 ± 37.1 | 132.7 ± 41.8 | 83.99 ± 2.65 | 28.21 ± 2.61 | 99.941 ± 0.050 | 23.99 ± 0.01 |

Original native centered Top-K SAEs, D=1536 and K=24. Mean ± sample SD across three encoder means, each averaging three SAE seeds. Shuffle reference is descriptive slice-label permutation, not a patient-level significance test. Supervised differs in supervision/augmentation and is a reference, not an isolated architectural ablation.
