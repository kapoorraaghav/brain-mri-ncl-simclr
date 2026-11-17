| Condition | Method | Accuracy (%) | Macro-F1 (%) | Balanced accuracy (%) | AUROC | Average precision |
| --- | --- | --- | --- | --- | --- | --- |
| Native | NCL | 84.84 ± 3.76 | 84.80 ± 3.75 | 84.85 ± 3.77 | 0.930 ± 0.027 | 0.940 ± 0.026 |
| Native | SimCLR | 87.92 ± 0.32 | 87.88 ± 0.33 | 87.93 ± 0.32 | 0.951 ± 0.005 | 0.959 ± 0.004 |
| Rank 8 | NCL | 80.84 ± 7.82 | 80.75 ± 7.80 | 80.86 ± 7.83 | 0.896 ± 0.059 | 0.901 ± 0.069 |
| Rank 8 | SimCLR | 76.43 ± 5.00 | 76.36 ± 4.97 | 76.44 ± 5.02 | 0.860 ± 0.049 | 0.871 ± 0.049 |
| Rank 8 + spectrum | NCL | 80.84 ± 7.82 | 80.75 ± 7.80 | 80.86 ± 7.83 | 0.896 ± 0.059 | 0.901 ± 0.069 |
| Rank 8 + spectrum | SimCLR | 76.43 ± 5.00 | 76.36 ± 4.96 | 76.44 ± 5.01 | 0.860 ± 0.049 | 0.871 ± 0.049 |
| 95% train variance | NCL | 75.57 ± 10.58 | 75.43 ± 10.58 | 75.58 ± 10.60 | 0.817 ± 0.123 | 0.814 ± 0.167 |
| 95% train variance | SimCLR | 86.68 ± 0.58 | 86.62 ± 0.60 | 86.71 ± 0.58 | 0.939 ± 0.008 | 0.950 ± 0.006 |
| 99% train variance | NCL | 75.93 ± 9.31 | 75.88 ± 9.25 | 75.94 ± 9.32 | 0.855 ± 0.078 | 0.861 ± 0.089 |
| 99% train variance | SimCLR | 87.06 ± 0.47 | 87.01 ± 0.49 | 87.08 ± 0.47 | 0.943 ± 0.006 | 0.954 ± 0.005 |

NB25 refitted raw-embedding probes, C=1 and max_iter=2000, training-only StandardScaler and transform. Mean ± sample SD across three encoders. No SAE-seed replications apply. Tumour-presence classification is slice-level, not diagnosis. These are the current two-method predictive results; do not mix them with the historical three-way probe values.
