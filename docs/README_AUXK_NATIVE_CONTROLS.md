# Native baseline versus AuxK

Open `22_auxk_native_controls_molab.py` in Molab. Local Jupyter alternative:
`22_auxk_native_controls.ipynb` with the `ncl_sae` kernel.

## Run on Molab

1. Import the new `.py` notebook and attach a CUDA GPU.
2. Set your read-only `HF_TOKEN` in the runtime secrets or `.env`, as before.
3. Keep the existing HF repository `YOUR_USERNAME/YOUR_PRIVATE_DATASET` and file
   `rank_control_embeddings_v1.zip`. The same 610 MB embedding bundle is reused;
   do not upload the large MRI PKL or a results ZIP as the input.
4. Keep the **new** output folder `auxk_native_controls_v1`. Do not select the old
   rank-control output folder; its immutable protocol is different.
5. Select encoder seed **10 only**, and both NCL and SimCLR. Apply settings.
6. Click **Download / verify embedding bundle**, then **Start / continue controls**.
   This starts **12 fits**: two methods, two SAE variants, three SAE seeds.
7. Wait until the worker finishes, click **Create full results ZIP**, refresh the
   Files panel and download the ZIP. It includes final dictionary weights.
8. Select **42 only**, apply settings and start again. Export and download again.
   Repeat for **52 only**. Same runtime/output folder accumulates all results;
   fresh runtimes contain only the runs actually completed there.

Thirty-six fits completes the full plan. Local Jupyter Run All selects all three
encoder seeds and starts the full plan; change `SELECTED_SEEDS` before running
if splitting sessions. The eight-hour budget is checked between fits. Resume in
the same output folder to reuse completed work; an incomplete fit restarts.

## Fixed comparison

Use the same 192-dimensional archived CLS embeddings, centered using training
means only. No PCA and no encoder training in this notebook. NCL and SimCLR both
use encoder seeds 10/42/52 and SAE seeds 10/42/52. D=1536, K=24, 30 epochs,
batch 512, Adam LR 0.0005, gradient clipping 1. The baseline is rerun alongside
AuxK with identical initial parameters and data order, rather than importing a
baseline produced by different code. All current encoder/rank results remain valid.

The AuxK arm adds positive Top-K dead-candidate reconstruction of a detached
main-path residual, without a decoder bias in that auxiliary reconstruction.
Candidates are units without a positive main activation for a full training pass.
Auxiliary activations do not reset the inactivity clock. K_aux=192, alpha=1/32,
threshold=138,399 processed training examples with the full input bundle. Ages
are tracked in examples including position within the minibatch, not epochs or
optimizer steps. The first dead-candidate event can occur at the end of pass one.

This is an MRI-adapted auxiliary-loss ablation, inspired by
[Gao et al., Appendix A.2](https://arxiv.org/html/2406.04093v1). Their full recipe
also changes initialization and decoder normalization. Those changes are excluded
from both arms here to isolate AuxK, and their much larger inactivity window is
not copied into this smaller study. Do not describe this as reproducing the full
Gao training recipe or as ruling out every possible dead-latent intervention.

## Results and interpretation

`all_control_fits.csv` contains all fit-level metrics and auxiliary-use counters.
`per_encoder_controls.csv` averages SAE seeds within encoder.
`control_summary.csv` reports mean, sample SD and count across complete encoder
groups. Require all three encoder groups before reporting the final comparison.
`auxk_minus_baseline.csv` compares variants within each method/encoder; the paired
NCL-minus-SimCLR file compares methods separately for each variant.

Distinguish validation alive counts, cumulative training alive counts and inactive
units at training end. They measure different things. Read `history.json` for
auxiliary batch counts, auxiliary loss, positive auxiliary activations, dead-unit
counts and main reconstruction loss. If AuxK never activates or has no positive
auxiliary activations, the intervention has not demonstrated effective mitigation.
No inactive candidates can legitimately mean zero auxiliary loss. Features whose
ReLU preactivations stay negative may remain unreachable with this variant.

Compare EV, eligibility coverage, purity minus its shuffle reference and the
one-slice-per-patient top-10 metric together. Auxiliary activity or more alive units
alone is not improved interpretability. Saved raw-embedding probes do not use SAE
codes and are identical between arms by design; they cannot assess utility changes
caused by the SAE. Slice-label shuffles are descriptive, not patient-level p-values.

Completed models save **once after the final epoch**, with transform, protocol,
training activity, environment and history. No per-epoch model writes. Completed
models can finish interrupted evaluation without retraining. Cloud storage remains
ephemeral: download the full results ZIP before closing it. These weights are SAE
weights, not the missing original encoder weights.

## Held-out evaluation audit

The audit is in `results/brats2023_expansion_audit_v1/heldout_input_audit.json`.
All six audited expanded NCL/SimCLR source archives have neither encoder weights
nor test embeddings. The six older local test caches have manifest
`c6c7ca4888ced1ee679942f2ab1e6f6750105d95d80efb012146b13b2f5c8304`, while the expanded
embedding bundle uses `eb3157a33d1240588e068e551ce352a97f39e009d09cec08ad288eb986fa6928`.
They cannot be silently substituted. Test labels or predictions were not evaluated.

Search surviving original cloud/runtime backups before considering encoder
retraining. If the original weights cannot be recovered, evaluating new images
requires a new registered encoder run with final encoder weights saved. Test
evaluation must freeze metrics and validation-only feature selection beforehand.
This notebook intentionally does not pretend to provide that missing test evidence.
Patient bootstrap and feature alignment are separate follow-up analyses, not outputs
of this notebook. Bootstrap uncertainty would describe the inspected validation
patients; it would not convert them into an unseen test cohort.

## Validation

Functional checks exercise small real-embedding subsets, gradients to inactive
candidates, no-candidate behavior, paired starts, final model saving/loading,
completed reuse, interrupted evaluation recovery and ZIP integrity. Evidence is
in `results/brats2023_expansion_audit_v1/auxk_notebook_checks.json` after checks pass.
The full 36-fit scientific experiment and live private-HF authentication are not
run during notebook construction.
