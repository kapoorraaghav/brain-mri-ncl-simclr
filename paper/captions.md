### S1_dataset

Patient-group and slice counts from the common split manifest. Only validation was evaluated; slices are clustered within patients.

### T1_native_SAE

Original native centered Top-K SAEs, D=1536 and K=24. Mean ± sample SD across three encoder means, each averaging three SAE seeds. Shuffle reference is descriptive slice-label permutation, not a patient-level significance test. Supervised differs in supervision/augmentation and is a reference, not an isolated architectural ablation.

### T2_probe_performance

NB25 refitted raw-embedding probes, C=1 and max_iter=2000, training-only StandardScaler and transform. Mean ± sample SD across three encoders. No SAE-seed replications apply. Tumour-presence classification is slice-level, not diagnosis. These are the current two-method predictive results; do not mix them with the historical three-way probe values.

### T3_geometry_selectivity

K=24, D=1536; mean ± SD across encoder means. Ranks/variance transforms are fitted on training only. Native and rank controls have fixed dimensions; retention controls choose the minimum dimension meeting a common threshold, with possible overshoot. Patient-distinct purity allows one slice per original patient among top ten. NCL PCA-rank and spectrum-matched conditions are mathematically identical; they are not independent confirmations. Retention does not equate semantic information.

### T4_AuxK

Matched native baseline/AuxK experiment; mean ± SD across three encoder means. AuxK alpha=1/32, auxiliary K=192 and inactivity window=138399 training examples. Initialization and decoder normalization are unchanged in both arms; this is not a reproduction of the complete Gao training recipe. Main-activation inactivity differs from validation alive counts. Auxiliary activation was observed, but NCL utilization did not improve.

### T5_feature_stability

Match eligible features by training activation correlation on 4096 patient-balanced training slices, then assess validation. Average three SAE-seed pairs within encoder, then mean ± SD across three encoders. Coverage denominator includes left training-eligible features; full directional coverage and 0.9 sensitivity are in supplements. Correlation is not semantic identity. Pairs share dictionaries and are not independent experiments.

### T6_paired_patient_intervals

1000 paired whole-patient bootstrap draws. SAE means are nested within encoder for selectivity; predictive outcomes have one probe per encoder. Intervals condition on fixed trained models and inspected validation patients, exclude training uncertainty and are not confirmatory significance tests. Top-ten/eligibility statistics are discontinuous; nominal interval coverage is not guaranteed. No cross-target best-result selection is used.

### S2_probe_detail

Exact per-encoder NB25 values (fractions, not percentages), confusion counts and solver flags; validation has 2373 negative and 2388 positive slices. All 30 fits stayed below the solver iteration limit.

### S3_K8_and_spatial

All K=8/K=24 original-run per-encoder means. Spatial Dice is CLS-to-patch transfer, not a patch-trained SAE or clinical segmentation model. Feature-selection and coarse-map limitations remain; do not compare best-feature maxima without accounting for candidate counts.

### S4_historical_three_way_probes

Historical three-way outputs; original probes, not NB25 refits. Supervised accuracy is available only here, not a fully matched NB25 F1/AUROC comparison. Small numerical differences from NB25 are retained rather than silently replaced; refit/source/software provenance differs and these should not be treated as identical predictions.

### F1_selectivity_utility

Selectivity and utility across all registered conditions. Points show encoder means (SAE seeds averaged for purity); error bars are sample SD across three encoders, not patient intervals. Small points expose individual encoders. Native/geometry SAE records and NB25 probes are separate outcomes.

### F2_auxk_response

Paired encoder-seed trajectories, each averaging three SAE seeds. AuxK raises SimCLR utilization while NCL utilization falls. This is the registered auxiliary-loss variant, not the complete Gao recipe; higher utilization alone is not a feature-quality claim.

### F3_feature_stability

Dots show encoder means over three SAE-seed pairs; crosses show random-matching references. Matching is training-only and assessed on validation. The 0.7 cutoff is descriptive, not a validated semantic threshold. Detailed coverage and 0.9 results remain in source data.

### F4_paired_probe_intervals

Paired whole-patient percentile intervals conditional on fixed encoder/probe fits and inspected validation patients. These are not encoder-seed SD, independent-test confirmation, multiplicity-adjusted tests or proof of population superiority.
