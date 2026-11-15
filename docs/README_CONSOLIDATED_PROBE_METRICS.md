# Notebook 25: consolidated probe metrics

Use `25_consolidated_probe_metrics_molab.py` in Molab or the matching `.ipynb`
locally with the `ncl_sae` kernel. CPU is sufficient; CUDA is not required.

Input: the existing `rank_control_embeddings_v1.zip` from private HF repository
`YOUR_USERNAME/YOUR_PRIVATE_DATASET`. Set the same read-only HF_TOKEN in secrets or `.env`.
No new HF upload, result ZIP or model-weight upload is needed. There is no
encoder or SAE retraining: only small logistic-regression probes are refitted.

Keep the fresh `probe_metrics_v1` output folder, all encoder seeds 10/42/52 and
both methods selected. Apply settings, Download / verify embedding bundle, then
Start / continue probes. There are **30 probes**: two methods, three encoder
seeds and five representation conditions. If splitting sessions, select one encoder
seed at a time (10 probes per seed), download results after each session, and
preserve the same output directory when storage survives. The full-result ZIP
must be downloaded from Files before closing an ephemeral runtime.

Conditions are native, PCA rank 8, rank-8 NCL-training-spectrum matching, training
PCA retention >=95%, and >=99%. All transform parameters are fitted to training
only. The <=192-dimensional PCA projection is embedded back into 192 coordinates,
matching the original probe geometry and StandardScaler. Do not standardize PCA
coordinates in a different basis: this can change regularization and the results.
Both retained-variance targets are reported, with their actual selected dimensions.
Native baseline and AuxK share raw backbone embeddings, so duplicate probes are
not counted as independent results. This notebook does not probe SAE codes and
does not include supervised encoders, which are absent from this input bundle.

The probe is StandardScaler followed by LogisticRegression(C=1,max_iter=2000),
with default unweighted classes and the existing decision rule. There is no new
threshold search, label rebalancing or validation-based model selection. Iteration
count and iteration-limit flags are saved; flagged fits need investigation before
their metrics are accepted. Probes are fit once on training and evaluated on val.

Reported metrics: accuracy, macro-F1 (both labels), balanced accuracy, tumour
precision/recall/F1, specificity, AUROC, average precision (the reported AUPRC),
and patient-macro accuracy. TN/FP/FN/TP provide confusion matrices for each probe.
Patient-macro accuracy averages slice accuracy per patient and then across
patients; it does not convert this task into patient-level diagnosis. The target
is binary slice tumour presence, not tumour subtype, grading or diagnosis.

Zero-denominator precision/F1 uses zero. Single-class bootstrap resamples have
undefined AUROC and balanced accuracy; no-positive resamples have undefined AP.
Missing values are retained, finite replicate counts are reported, and interval
bounds are omitted if fewer than 95% of replicates are finite. Check class counts
and prevalence alongside AP and the confusion counts. AP is not trapezoidal PR AUC.

One thousand whole-patient bootstrap draws are shared by every probe. Every drawn
patient contributes all their slices, including multiplicity. Metrics are
recomputed with those sample weights. Encoder means/SD describe three encoders,
not nine SAE runs. Across-encoder bootstrap intervals require all three encoders;
paired NCL-minus-SimCLR differences use the same patient draws. These percentile
intervals condition on the trained encoders/probes and are exploratory because
validation data were already inspected. They do not include training uncertainty,
establish clinical performance or supply missing held-out/external evidence.

Main outputs:

- `probe_metrics.csv`: every method/encoder/condition, confusion counts and solver flags.
- `probe_summary.csv`: encoder mean, sample SD and count for each score.
- `patient_bootstrap_intervals.csv`: full-three-seed score and paired-difference intervals.
- Per-case `predictions.csv`: sample/patient identity, truth, prediction and tumour probability.
- Per-case `patient_bootstrap.csv`: all resampled score values and confusion counts.
- Per-case `probe_and_transform.npz`: coefficients, intercept, classes, standardizer and transform.
- Transform diagnostics, patient draw indices, patient order, protocol and environment.

Full exports include these files and fitted probe parameters; the input cache and
HF credentials are excluded. Completed probe cases can be reused with file-hash
checks. Stopping finishes the current probe and bootstrap. A partially completed
case is refit on the next run. Download the entire ZIP, not only the summary CSV.

Construction checks use small real-embedding probes and compare weighted metrics
against sklearn, including single-class bootstrap behavior and all five transforms.
See `results/brats2023_expansion_audit_v1/probe_metrics_notebook_checks.json`.
The complete 30-probe evaluation and live private-HF authentication were not run
during construction. Existing accuracy values should be reconciled with the new
predictions; differences across software versions or convergence must be reported.
