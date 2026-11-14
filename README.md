# Auditing Feature Populations and Label Selectivity in Brain MRI

**Author: Raaghav Kapoor.** Unpublished research draft, 2026.

This project audits how SAE feature populations, annotation context and prediction
representations affect label-selectivity comparisons in brain tumour FLAIR MRI.
It compares existing NCL and SimCLR objectives and a supervised reference; it does
not introduce a new contrastive method, SAE architecture or clinical explanation.

## Publication progress

The repository is being published in five planned batches. **This first batch
contains setup, preprocessing, the shared runtime and historical training code.**
The audit workflows, current manuscript and aggregate results belong to later
batches. Their paths below are explicitly marked as planned, not executable
entry points in this partial release. See [docs/WORKFLOW.md](docs/WORKFLOW.md).

## Setup and checks available now

Use Python 3.11 or newer. Install PyTorch appropriate for your hardware, then:

```bash
python -m pip install -r requirements.txt
python scripts/run_embedding_experiment.py --help
python scripts/run_embedding_experiment.py --stage rank --dry-run
python scripts/run_embedding_experiment.py --stage probe --dry-run
```

The dry-run commands inspect the plan without loading inputs, fitting models or
producing scientific results. Real execution needs authorized inputs and their
identity receipts. Receipts and stage-specific extensions are planned for the
second batch; do not interpret a successful dry-run as a complete installed study.
Training and notebook extras are in `requirements-training.txt`. Additional
audit and preservation requirements are declared for the later batches.

## Entry-point map

| Workflow | Entry point | Status after the first batch | Required inputs |
|---|---|---|---|
| Patient-grouped preprocessing | `notebooks/13b_brats2023_preprocessing_fixed.ipynb` | Available; review configuration before execution | Authorized BraTS images, original patient mapping and earlier split metadata |
| Historical image training | `notebooks/historical_training/20_*` | Available as an archive | Trusted indexed preprocessed PKL |
| Earlier general training CLI | `scripts/revision_experiments.py` | Available; defaults do not define the final historical experiment | Processed metadata and matching configuration |
| Shared embedding controls | `scripts/run_embedding_experiment.py` | Help and dry-run available; full stages need later files | Identity-checked embedding bundle; model archive for patient stage |
| Preserved encoder runs | `notebooks/30_*` and `31_restore_preserved_saes_molab.py` | Planned for batch 2 | Authorized processed PKL and corresponding preserved exports |
| Rank, AuxK, stability, retained variance and probes | `notebooks/21_*` through `25_*` | Planned for batch 2 | Historical embedding bundle; AuxK model archive for stability |
| Current feature-population audit | `scripts/feature_population_audit.py`, `notebooks/32_feature_population_audit.ipynb` | Planned for batch 2 | Historical embeddings, matching SAE states and private validation covariates |
| Patient-resampling audit diagnostics | `scripts/bootstrap_feature_budget_patients.py`, `bootstrap_null_correlations.py` | Planned for batch 3 | Saved per-slice activations and patient identities; not distributed |
| Current manuscript and its figures | `paper/current/` | Planned for batch 5 | No patient input needed to read or compile the manuscript |

## Scientific scope and provenance

The expanded training cohort has 1,059 patient groups. The reported analyses
reuse 37 validation patients involved in checkpoint selection. The registered
test patients were exposed in earlier experiments and are not fresh confirmation.
SAE fits are nested within three encoder seeds. Slice counts and dictionary units
are not independent patient samples.

Historical encoder weights for the expanded experiment remain unavailable. A
separate preservation rerun retained its own models and verified inference; it
does not recover or replace the historical models. Existing legacy files in this
repository's `checkpoints/`, `notebooks/` and `results/` predate the expanded audit
and must not be silently substituted for its registered inputs.

The historical notebooks retain their original filenames for traceability.
Despite `checkpointed` in those names, they did not preserve the encoder weights
needed to reproduce the expanded historical image inference. They are archived
code, not the recommended interface for a new preserved training run.

## Data, credentials and release status

This batch adds no raw MRI datasets, private embedding archives or new model
weights. Obtain authorized inputs under the applicable BraTS terms and consult
[data/README.md](data/README.md). Keep credentials in environment variables; the
tracked `.env.example` contains an empty placeholder. Only load trusted PKLs.

See [CITATION.md](CITATION.md), [LICENSE_NOTICE.md](LICENSE_NOTICE.md) and
[PUBLICATION_NOTES.md](PUBLICATION_NOTES.md). A software license has not been
selected. The public repository is author-identifying; its URL is not an
anonymous supplementary artifact for double-blind review.
