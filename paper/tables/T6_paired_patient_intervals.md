| Outcome | Condition | Metric | NCL minus SimCLR (points) | Patient percentile interval |
| --- | --- | --- | --- | --- |
| Selectivity | Baseline | mean_purity | 4.11 | [1.02, 5.85] |
| Selectivity | Baseline | patient_distinct_purity | 5.47 | [2.57, 6.36] |
| Selectivity | AuxK | mean_purity | 3.40 | [0.47, 5.23] |
| Selectivity | AuxK | patient_distinct_purity | 4.80 | [1.57, 5.30] |
| Prediction | Native | macro_f1 | -3.08 | [-4.61, -1.67] |
| Prediction | Rank 8 | macro_f1 | 4.39 | [2.75, 5.86] |
| Prediction | Rank 8 + spectrum | macro_f1 | 4.39 | [2.75, 5.87] |
| Prediction | 95% train variance | macro_f1 | -11.19 | [-13.61, -8.73] |
| Prediction | 99% train variance | macro_f1 | -11.13 | [-13.73, -8.49] |

1000 paired whole-patient bootstrap draws. SAE means are nested within encoder for selectivity; predictive outcomes have one probe per encoder. Intervals condition on fixed trained models and inspected validation patients, exclude training uncertainty and are not confirmatory significance tests. Top-ten/eligibility statistics are discontinuous; nominal interval coverage is not guaranteed. No cross-target best-result selection is used.
