# Rank/covariance controls for the expanded BraTS experiments

Use this follow-up to test whether the observed NCL–SimCLR selectivity difference
persists after controlling representation dimensionality and covariance spectrum.
The encoders are **not retrained**. Inputs come from the six completed NCL/SimCLR
train/validation embedding exports. Raw MRI files and the 12 GB PKL are not needed.
The original nine runs, archives and results are preserved.

## Files

- Molab/marimo: `21_rank_covariance_controls_molab.py` (self-contained).
- Local Jupyter: `21_rank_covariance_controls.ipynb` (self-contained; ncl_sae kernel).
- Input bundle: `data/rank_control_embeddings_v1.zip`, about **610 MB**.
- Bundle SHA256: `e41d4ae24932127a5747073d7225bdfd2d20022ca8a22027222674b7494cf9eb`.

The input ZIP contains only verified train/validation CLS embeddings, their sample
identities, the shared split manifest and original validation probe references.
No encoder/SAE weights or test features are included. The split manifest includes
test metadata solely to retain the original partition identity; no test embeddings
or images are read. Each embedding file is checked against its SHA256 and sample
ordering, including on later use from the extraction cache.

## Primary experiment

For each encoder seed 10/42/52, compare NCL and SimCLR under three conditions:

1. **Native representation:** new centered baseline SAEs on the original frozen
   embeddings. This recreates a reusable baseline because the original dictionaries
   were not saved as model weights.
2. **PCA rank 8:** fit PCA on training embeddings only, retain eight components,
   and project back into the original 192 input coordinates. No whitening/rescaling.
3. **Training-spectrum matched, rank 8:** start from the same train-only PCA basis,
   then scale components to the top-eight NCL training eigenvalues for the same
   encoder seed. Both methods therefore target the same retained training spectrum,
   trace, effective rank and participation ratio. The implementation verifies the
   resulting training eigenvalues rather than merely labeling them matched.

Because NCL itself supplies the reference eigenvalues, its PCA-rank and
spectrum-matched transforms are mathematically the same. They are retained as
separate labeled cases for a uniform paired output layout, not independent
confirmations of the NCL result. With a fixed SAE seed, their outcomes should match.

Use three SAE seeds 10/42/52 for each condition. Defaults: full 138,399 training
slices, dictionary 1536, K=24, 30 epochs, batch 512, Adam LR 0.0005, gradient clip 1.

**Full experiment: 2 methods × 3 encoder seeds × 3 conditions × 3 SAE seeds = 54 fits.**
One selected encoder seed runs **18 fits**, including both methods.
Primary rank 8 is fixed before these control outcomes. Optional ranks 4/24 are
separate sensitivity experiments requiring an updated plan and a fresh output
directory, not an automatic search for the best validation result.

Only baseline Top-K SAEs are fitted. An AuxK comparison is a subsequent control;
do not describe this notebook as having completed that experiment.

## Run on Molab

1. Upload **`rank_control_embeddings_v1.zip`** from the project's `data` folder to
   the root of your existing **private** HF repository `YOUR_USERNAME/YOUR_PRIVATE_DATASET`.
   Do not extract it before uploading. The Molab browser-upload limit makes HF
   preferable for this 610 MB input.
2. Upload `21_rank_covariance_controls_molab.py` as a new Molab notebook and attach
   a CUDA GPU. Its runner, reader and worker code are embedded; no helper scripts
   or project checkout are required.
3. Reuse your existing read-only HF token. Add `HF_TOKEN` to this runtime's secrets
   or `.env` beside `notebook.py`. Do not paste the token in code or this chat.
4. Keep both methods selected. Default encoder seed is **10** (18 SAE fits).
   Apply settings and click **Download / verify embedding bundle**.
5. Click **Start / continue controls**. Training runs in a background process;
   the progress display and metric download remain responsive.
6. Wait for worker status **completed selected controls**. Click **Create full
   results ZIP**, refresh Files and download the generated `_ALL_RESULTS.zip`.
   Its name identifies the encoder seeds present. It contains the new SAE models,
   transform recipes, detailed results and provenance, and excludes the large input
   extraction cache.
7. Run encoder seeds **42** and **52** too. In a runtime that retains the results
   folder, changing only execution seed selection reuses completed fits under the
   same immutable scientific protocol. In fresh runtimes, use the same input ZIP
   and token, and download each seed's output ZIP before closing.

The full-plan progress denominator is 54; a single-seed session completes 18.
Across three fresh runtimes, send all three result ZIPs for the paired comparison.

## Local Jupyter

Open `21_rank_covariance_controls.ipynb`, select `ncl_sae`, then **Run All** to
execute the full plan. `SELECTED_SEEDS=[10,42,52]` by default; change it before
starting if splitting runs. It finds the local input bundle automatically from
the project or notebook directory and writes `results/rank_covariance_controls_v1`.
No HF upload or token is needed for this local route.

## Saving, stopping and continuation

**No per-epoch model checkpoints or cloud model uploads.** Each fit writes small
JSON training histories and saves one final `completed_model.pt` after training.
That file includes the SAE weights, PCA/scale recipe, protocol, training history
and training environment. It contains no optimizer/scaler/RNG training-resume state.
The dictionaries support safe `torch.load(..., weights_only=True)` loading.

- **Stop after completed fit** waits until the current training/evaluation finishes.
  Completed dictionaries are reused with protocol and model-file hash checks.
- An interrupted incomplete fit must retrain from epoch 1. If training finished
  and the final model saved before evaluation was interrupted, evaluation can be
  completed from that model without retraining.
- The default time budget is 8 hours, checked between fits. One fit may extend
  beyond that budget. Choose a budget below your provider's actual runtime limit.
- Cloud scratch can still disappear. Download the full output ZIP before closing.
  To continue completed work in a fresh runtime, restore its contents inside the
  chosen output directory before starting with the same input bundle and plan.
- Never run two workers against the same output directory. A process lock rejects
  duplicate launches in one runtime.

The final models are **new control SAE dictionaries**, not recovered MRI encoders.
They do not enable image-level test/external evaluation without encoder weights.

## What is reported

- PCA-only retained validation information relative to the original training mean.
- SAE variance explained in the transformed representation.
- Full original-coordinate fidelity after undoing retained-component scaling and
  projecting back. This separates PCA information loss from additional SAE error.
- Actual training and validation effective rank, participation ratio and total
  variance. Spectrum matching is imposed **only on training data**; validation
  spectra are measured rather than forced to match.
- A fresh C=1 standardized logistic probe fitted on training features only for
  each representation, with validation accuracy. This checks changes in task
  information after projection, not just ease of reconstruction.
- Alive/eligible feature counts, top-10 purity, entropy and slice-label permutation
  reference, matching the main experiment's metric definitions.
- Secondary top-10 purity using at most one activating slice per distinct patient,
  with eligible-feature coverage and its own descriptive shuffle reference.
- Per-SAE results, means across SAE seeds within encoder, and mean/SD across
  complete encoder groups. Partial SAE groups are excluded from the across-encoder
  summary. A paired NCL-minus-SimCLR table matches encoder seed and condition.

The user-facing CSVs are `all_control_fits.csv`, `per_encoder_controls.csv`,
`control_summary.csv` and `paired_control_differences.csv`. Each condition also has
transform diagnostics, a validation probe, feature tables and completed models.

## Interpretation

These controls can test robustness **under the specified transforms**. Equal PCA
dimension alone does not equal equal effective rank. Spectrum matching goes further,
but it does not match eigenvectors, semantics, nuisance information or optimization
geometry. Compression may discard useful SimCLR information; inspect PCA retention,
original-coordinate reconstruction and the probe alongside purity. Small absolute
MSE after variance scaling is not proof of improved reconstruction quality.

If the purity gap shrinks after matching, that is evidence that dimensionality/
spectrum contributes under these transformations; it is not proof that rank alone
caused the original difference. If the gap persists, that is evidence of robustness
under this control, not proof of causal disentanglement or clinical interpretability.

Patient-distinct top-10 examples address repeated slices from one patient, but their
slice-label permutation is not a formal patient-level significance test. Eligibility
and class imbalance still matter. This notebook does not calculate spatial Dice.

All results remain exploratory validation follow-ups on the same previously
inspected 37 patients. They are not new test or external-cohort evidence. Do not
treat three SAE seeds as three independent encoder experiments.

## What related papers say

The cited review is in [the related-work note](../docs/RANK_SAE_RELATED_WORK_2026.md),
with a draft paragraph for the manuscript and [BibTeX references](../docs/rank_sae_related_work.bib).
It covers nine primary papers, checked 26 October 2026, and distinguishes preprints
from verified conference records. Both notebooks include a short linked overview.

The central motivation is that dimensional collapse, covariance, dead-latent
optimization and initialization can all affect an SAE comparison. The original
NCL paper also distinguishes projector interpretation from backbone probing;
our backbone-CLS SAE results need that qualification. These sources motivate
controls rather than establish that our MRI finding is independent of rank.
The current experiment does not implement AuxK or reproduce the cited benchmarks.

## Verification performed during construction

The real training-spectrum construction is checked on all six full embedding sets.
GPU smoke tests use small real-embedding subsets to check training, metrics,
inverse-transform fidelity, one-time final model saving/loading, completed-model
reuse, interrupted-evaluation recovery, background execution and duplicate-worker
protection. These are not full 54-fit scientific results. The private HF download
is not live-tested against your account. Evidence is saved in
`results/brats2023_expansion_audit_v1/rank_control_notebook_checks.json` after tests pass.
