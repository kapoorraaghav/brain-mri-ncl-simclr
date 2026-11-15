# Patient uncertainty, dictionary stability and retained-variance controls

These are separate follow-ups to the completed rank-control and AuxK experiments.
Existing result archives remain unchanged. No encoder training is performed.
All analyses concern the expanded cohort's previously inspected validation patients.

## Notebook 23: patient bootstrap and feature stability

Molab: `23_patient_bootstrap_feature_stability_molab.py`.
Local Jupyter: `23_patient_bootstrap_feature_stability.ipynb`.

Inputs:

- Existing `rank_control_embeddings_v1.zip` (610 MB), already in the private
  `YOUR_USERNAME/YOUR_PRIVATE_DATASET` HF repository.
- **Additional input:** `auxk_native_controls_v1_enc10-42-52_ALL_RESULTS.zip`
  (79.7 MB), containing the 36 verified native baseline/AuxK dictionaries.
  Upload that completed ZIP once to the root of the same private HF repository,
  keeping its exact filename. Do not upload the incomplete one/two-seed archives.
  You can alternatively place it directly in the Molab runtime and set the
  models-path field to that file; an existing matching file is reused before HF
  download. The archive must be present in each separate runtime that uses it.

The source-model SHA256 is
`724f0a4e5d81b1bcebca019bdf17bb1eeb673ce7d1d8640960909e82fd165856`.
The embedding SHA256 remains
`e41d4ae24932127a5747073d7225bdfd2d20022ca8a22027222674b7494cf9eb`.
Archive checks and each model's recorded hash are verified before evaluation;
model manifest and encoder/SAE identity are checked against the input bundle.
Loading uses `torch.load(..., weights_only=True)`.

Attach a CUDA GPU, keep your read-only HF_TOKEN in runtime secrets or `.env`, and
import the new notebook. Leave all three encoder seeds and both methods selected.
Use the new `patient_stability_v1` output folder. Apply settings, verify/download
inputs, then start. This performs 36 **saved-model analyses**, not 36 training runs.
Use Create full results ZIP after the worker stops; the export includes analysis
tables and replicates, not the large input models or embedding cache. Keep the
original model archive separately. Local Jupyter can find the model archive in
the project's `data` folder or the user's Downloads folder.

### Patient bootstrap definitions

The resampling unit is the patient. Draw 37 patient IDs with replacement 1,000
times, including all slices for every drawn patient. The exact same draws are
used for every method, encoder, SAE seed and variant, making differences paired.
Resampling individual slices would incorrectly treat correlated slices as
independent observations. There is no stratification by observed outcomes.

For each replicate, recompute positive support, eligible features and dominant
label purity among the top 10 activating slices. Per-patient top-10 candidate
pools yield the same statistic as explicitly duplicating all slices, while
avoiding a full sort of every slice in every replicate. Also recompute validation
reconstruction EV, alive and eligible feature counts. The denominator for EV uses
the fixed training mean, matching the original experiment; it is not refitted to
each resample. Ties are implementation-dependent, so point-estimate differences
from archived purity are saved explicitly for inspection.

Patient-distinct top-10 uses at most one slice per original patient ID. Repeated
copies of an ID do not create new distinct identities. Feature eligibility and
coverage therefore change when fewer unique patients occur in a resample. Never
average undefined replicates silently: interval tables report finite and total
replicate counts, and suppress bounds if more than 5% are undefined.

SAE seeds are averaged within encoder; encoder means are then averaged equally.
Shared draws give NCL-minus-SimCLR and AuxK-minus-baseline intervals. The across-
encoder summary requires all three encoder groups. Encoder seeds are **fixed**
for this calculation: intervals do not include encoder training variation,
hyperparameter selection or an independent external cohort. Across-encoder SD is
a separate description and three SAE seeds are not three new patient samples.

Intervals use empirical 2.5th/97.5th percentiles, not p-values or calibrated
significance claims. Top-10 selection and eligibility are discontinuous, and
duplicated patients can shift extreme-example statistics. With only 37 inspected
validation patients, nominal 95% coverage is not guaranteed. Report interval
width, eligibility and point-estimate behavior; do not treat exclusion of zero
as confirmatory evidence. No permutation-adjusted-purity intervals, clinical
outcomes or spatial Dice are generated here.

For alive/eligible feature counts, treat the percentile bounds as **resampling
sensitivity ranges**, not confidence bounds on a population concept count.
Rare features disappear when patients are omitted, so these bounds can sit below
the observed count. The paper's uncertainty analysis should focus on the score
and paired-difference estimates, with these limitations visible.

Outputs: `patient_bootstrap_intervals.csv`, `bootstrap_fit_points.csv`,
`patient_bootstrap_draw_indices.csv`, `patient_order.json`, and the per-fit
`bootstrap_replicates.csv` files. The saved indices reproduce every patient draw.

### Feature stability definitions

Within each method, encoder and variant, compare the three pairs of SAE seeds.
Do not directly compare coordinate directions across independently trained
encoders: their bases need not align. The 4,096 alignment examples are selected
from **training only** by patient-balanced round robin without labels. The same
sample identities are used for all methods and models and are saved in
`alignment_training_samples.csv`.

Features need >=10 positive examples in that training subset and nonzero training
variance. Compute activation correlations and use one-to-one maximum-weight
assignment on the training matrix. The assignment is then frozen. Report matched
activation correlation and positive-support Jaccard on validation activations,
decoder cosine, eligible/matched counts, and a random one-to-one matching reference.
The matching algorithm's rectangular assignment leaves surplus features unmatched;
coverage denominators include eligible features on each side separately.

Constant validation features have undefined correlation and stay visible in the
variable-match count. Thresholds 0.7 and 0.9 are explicitly descriptive cutoffs;
report both, not the more favorable one. Means are over all matched eligible
features, not only successful matches. Random matching is a descriptive reference,
not a hypothesis test. Similar activation directions need not correspond to
clinically meaningful or causally disentangled concepts.

Outputs: per-pair `matched_features.csv`, `feature_stability_pairs.csv`,
`feature_stability_per_encoder.csv` and `feature_stability_summary.csv`. Average
the three pairs within encoder before interpreting across-encoder mean/SD; the
pairs share dictionaries and are not independent replications.

The implementation uses [SciPy's assignment routine](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html).
The shared-index resampling principle corresponds to the paired-bootstrap
definition in the [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html);
this notebook explicitly implements patient clusters rather than passing slices
as independent observations to that function.

## Notebook 24: comparable training-variance retention

Molab: `24_retained_variance_controls_molab.py`.
Local Jupyter: `24_retained_variance_controls.ipynb`.

This notebook needs **only** the existing embedding ZIP and the same HF_TOKEN.
It does not require the completed AuxK model ZIP. Use a fresh
`retained_variance_controls_v1` output folder and CUDA GPU.

Select encoder **10 only**, keeping both methods. Apply settings, verify/download
inputs, then start. This runs **12 new SAE fits**: two methods, two variance
targets and three SAE seeds. Export full results with final models; then repeat
for 42 and 52. A shared output folder accumulates 36 fits. Local Jupyter defaults
to all three encoder seeds; edit SELECTED_SEEDS before Run All if splitting runs.

PCA is fitted to training embeddings only. Choose the smallest dimension whose
training covariance eigenvalues explain at least **95%** of training variance
(primary). Repeat at **99%** (registered secondary). Both targets must be reported;
do not select one after inspecting purity. Discrete eigenvalues can overshoot the
target, particularly for low-rank NCL: actual fractions are reported. This is
comparable retention by a common threshold, not exact matching of fractions.

Dimensions selected from the full training inputs:

| Encoder seed | NCL, 95% | SimCLR, 95% | NCL, 99% | SimCLR, 99% |
|---|---:|---:|---:|---:|
| 10 | 8 | 74 | 11 | 95 |
| 42 | 3 | 60 | 4 | 90 |
| 52 | 3 | 90 | 6 | 121 |

These are training-spectrum audit results, **not completed SAE outcomes**.
Projection back to 192 coordinates preserves the existing SAE input width.
No whitening, spectrum matching or AuxK is added. Keep D=1536, K=24, 30 epochs,
batch=512, Adam LR=0.0005, gradient clipping=1 and the same baseline initialization.
Existing native baselines need not be trained again.

Comparable training variance does not guarantee equal predictive information,
validation retention, effective rank, eigenvectors or semantic content. Report
actual train and validation retention, selected rank, PCA-only EV, original-
coordinate and transformed-coordinate SAE EV, validation probe and feature
selectivity. Compare with existing native and rank-8/spectrum controls; do not
replace them. Summaries group by target even when dimensions differ between
methods or encoder seeds. A similar variance fraction is a sensitivity control,
not a causal intervention isolating semantics.

Final SAE weights save once per completed fit, along with transform and protocol.
No per-epoch checkpoints. Incomplete fits restart; completed fits can be reused.
Download full results ZIP before closing the runtime.

## Checks and scope

Checks include literal-versus-pooled bootstrap agreement, repeated-patient identity
handling, known feature-permutation recovery, all-six-input rank selection, small
real-embedding fits, actual saved-model analysis, complete-group guards, saved
model reuse, export integrity, notebook schema and marimo validation.
Evidence: `results/brats2023_expansion_audit_v1/followup_notebook_checks.json`.
The local full-analysis result directory, when complete, is
`results/patient_stability_v1`. Retained-variance scientific SAE fits are separate
from construction smoke tests. Private HF authentication is not live-tested.

Neither notebook restores the missing expanded-run encoder weights, evaluates
the held-out test, establishes external generalization or replaces clinical
validation. The known held-out-input limitation remains documented separately.
