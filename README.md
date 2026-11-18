# Auditing Feature Populations and Label Selectivity in Brain MRI

**Author: Raaghav Kapoor.** Unpublished research draft, 2026.

This project audits how SAE feature populations, annotation context and prediction
representations affect label-selectivity comparisons in brain tumour FLAIR MRI.
It compares existing NCL and SimCLR objectives and a supervised reference; it does
not introduce a new contrastive method, SAE architecture or clinical explanation.

## Publication progress

All five release batches are available. The current manuscript and figures are in
[paper/current](paper/current/README.md). See [docs/WORKFLOW.md](docs/WORKFLOW.md)
for entry points and input requirements.

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
identity receipts. Receipts and stage-specific control extensions are now present;
a successful dry-run still does not load or verify your private input archives.
Training and notebook extras are in `requirements-training.txt`. Additional
audit and preservation dependencies are in their corresponding requirements files.

## Entry-point map

| Workflow | Entry point | Availability in published batches | Required inputs |
|---|---|---|---|
| Patient-grouped preprocessing | `notebooks/13b_brats2023_preprocessing_fixed.ipynb` | Available; review configuration before execution | Authorized BraTS images, original patient mapping and earlier split metadata |
| Historical image training | `notebooks/historical_training/20_*` | Available as an archive | Trusted indexed preprocessed PKL |
| Earlier general training CLI | `scripts/revision_experiments.py` | Available; defaults do not define the final historical experiment | Processed metadata and matching configuration |
| Shared embedding controls | `scripts/run_embedding_experiment.py` | Available; full stages require authorized input archives | Identity-checked embedding bundle; model archive for patient stage |
| Preserved encoder runs | `notebooks/30_*` and `31_restore_preserved_saes_molab.py` | Available; review input requirements | Authorized processed PKL and corresponding preserved exports |
| Rank, AuxK, stability, retained variance and probes | `notebooks/21_*` through `25_*` | Available; review input requirements | Historical embedding bundle; AuxK model archive for stability |
| Current feature-population audit | `scripts/feature_population_audit.py`, `notebooks/32_feature_population_audit.ipynb` | Available; review input requirements | Historical embeddings, matching SAE states and private validation covariates |
| Patient-resampling audit diagnostics | `scripts/bootstrap_feature_budget_patients.py`, `bootstrap_null_correlations.py` | Available; requires private patient-level inputs | Saved per-slice activations and patient identities; not distributed |
| Current manuscript and its figures | `paper/current/` | Available | No patient input needed to read or compile the manuscript |

## Inspecting preserved exports without loading weights

```bash
python scripts/inspect_preserved_archive.py path/to/export.zip
python scripts/inspect_preserved_archive.py path/to/export.zip --require-saes
python -m unittest discover -s tests -p test_preserved_archive_inspection.py -v
```

The inspector reads ZIP directory metadata only. It lists checkpoints/manifests
and missing expected names; it does not extract files, execute archived code,
unpickle weights, verify checkpoint hashes or establish completed training.
Exit codes are 0 for expected names present, 2 for missing names and 1 for
unreadable or unsafe metadata. `--require-saes` additionally checks native
baseline SAE filenames for seeds 10, 42 and 52.

## Historical audit preflight

```bash
python scripts/check_audit_inputs.py --dry-run
python scripts/check_audit_inputs.py
```

The first command checks paths only; the second verifies the three registered
historical input hashes. Neither loads a model or writes predictions. Missing
private inputs are expected in a clean checkout and return status 2. See
[docs/FEATURE_POPULATION_AUDIT.md](docs/FEATURE_POPULATION_AUDIT.md) for overrides,
scope and the distinction between the historical and preserved model families.

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

The five release batches add no raw MRI datasets, private embedding archives or new model
weights. Obtain authorized inputs under the applicable BraTS terms and consult
[data/README.md](data/README.md). Keep credentials in environment variables; the
tracked `.env.example` contains an empty placeholder. Only load trusted PKLs.

See [CITATION.md](CITATION.md), [LICENSE_NOTICE.md](LICENSE_NOTICE.md) and
[PUBLICATION_NOTES.md](PUBLICATION_NOTES.md). A software license has not been
selected. The public repository is author-identifying; its URL is not an
anonymous supplementary artifact for double-blind review.
