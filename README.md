# Auditing Feature Populations and Label Selectivity in Brain MRI

Code and aggregate results for a controlled empirical study of sparse autoencoders
(SAEs) on NCL and SimCLR brain-MRI representations. This is a research draft, not
an accepted paper or a clinical diagnostic tool. The contribution is the empirical
selectivity-utility trade-off, not a new NCL or SAE algorithm.

## Current findings and paper

This repository contains the earlier geometry/control experiments and the later
feature-population audit. The current manuscript is `paper/current/main.pdf`;
its LaTeX, 20-reference bibliography and figures are in the same directory.
The October synthesis in the other `paper/` files is retained as historical
context, not the current manuscript or its final claim wording.

NCL's all-eligible raw purity lead is 4.11 percentage points. At J=8 its
usage-ranked galleries are less pure than SimCLR's (gap -8.19 points;
conditional patient resampling envelope [-12.50, -1.25]). Mean comparisons
depend on feature budget; larger-budget envelopes span zero. At J=64,
usage-ranked purity is 0.98 points below NCL's random-subset mean and 3.25
points above SimCLR's. The stratified context reference removes a stable
NCL lead. Native and full-code probes favor SimCLR; selected-code probes
favor NCL. These are exploratory results on the reused validation cohort,
not independent patient confirmation, a causal geometry finding or a
demonstration of clinical interpretability.

See `docs/FEATURE_POPULATION_AUDIT.md` and
`results/feature_population_audit_v1/` for the latest audit procedures and
aggregate results. `scripts/feature_population_audit.py` and the accompanying
verification, sensitivity and bootstrap scripts implement those analyses.
`notebooks/32_feature_population_audit.ipynb` provides their notebook entry point.

## Setup

Use Python 3.11 or newer. Install PyTorch appropriate to your hardware following
its official instructions, then install the analysis dependencies:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/run_embedding_experiment.py --help
python scripts/run_embedding_experiment.py --stage probe --dry-run
```

Training/preprocessing/Molab extras: `python -m pip install -r requirements-training.txt`.
Use a CUDA GPU for SAE training; CPU is sufficient for logistic probes. Compatibility
ranges are provided, not a guaranteed cross-platform lockfile. Recorded final
probe environment: Python 3.13.11, torch 2.11.0+cu130, NumPy 2.4.6, pandas 3.0.3,
scikit-learn 1.9.0. Small source/refit differences from historical probes are
disclosed in the paper evidence.

## Inputs and data access

This release does not include raw MRI datasets, embeddings, model archives or private storage
locations. Obtain authorized BraTS-derived inputs under their applicable terms.
Exact result reproduction needs the original identity-checked embedding bundle
`data/rank_control_embeddings_v1.zip`. Its SHA256 is recorded in `configs/`.
Patient-bootstrap/feature-stability reproduction additionally needs the completed
36-model AuxK archive; its filename/checksum are in the second receipt. Providing
a different dataset is a new experiment, not reproduction of the published values.
See `data/README.md` and `docs/WORKFLOW.md`.

For Molab, import notebooks 21-25, set your own HF repository in the UI and set
read-only `HF_TOKEN` in runtime secrets. No token is embedded in this repository.
The placeholder `YOUR_USERNAME/YOUR_PRIVATE_DATASET` must be replaced by your own
authorized source. Local source mode is also supported. Do not upload data/model
archives to GitHub. Preserve result ZIPs before closing cloud runtimes.

## Embedding experiment CLI

```bash
python scripts/run_embedding_experiment.py --stage rank --bundle data/rank_control_embeddings_v1.zip --encoder-seeds 10 --output results/rank_v1
python scripts/run_embedding_experiment.py --stage auxk --bundle data/rank_control_embeddings_v1.zip --encoder-seeds 10 --output results/auxk_v1
python scripts/run_embedding_experiment.py --stage retention --bundle data/rank_control_embeddings_v1.zip --encoder-seeds 10 --output results/retention_v1
python scripts/run_embedding_experiment.py --stage patient --bundle data/rank_control_embeddings_v1.zip --models data/auxk_native_controls_v1_enc10-42-52_ALL_RESULTS.zip --output results/patient_v1
python scripts/run_embedding_experiment.py --stage probe --bundle data/rank_control_embeddings_v1.zip --output results/probe_v1
```

Full plans: rank 54 SAE fits; AuxK 36; retention 36; patient analysis 36 saved models;
probe 30 small fits. Defaults select all three encoder seeds. For split sessions,
run 10, then 42, then 52 in the same output directory when storage persists. SAE
stages save a final dictionary once per completed fit, without per-epoch recovery.
Incomplete fits restart. Stop/time budgets are checked between completed units.
CLI export writes a ZIP; result folders have immutable input/plan/code registration.

## Repository map

- `src/`: exact core runtime and appended experiment-specific extensions.
- `notebooks/21-25`: self-contained Molab and local Jupyter workflows.
- `notebooks/historical_training/`: the nine exact historical training workflows,
  with only storage configuration sanitized. Despite their filenames, these
  versions **did not retain encoder/SAE weights**. They are not recommended as a
  new checkpointed training interface; read the workflow notes first.
- `notebooks/13b...`: provenance-specific patient-grouped preprocessing. It needs
  the original mapping and earlier split/metadata inputs; not a one-click raw-data recipe.
- `scripts/revision_experiments.py`: general earlier checkpointed CLI; its defaults
  are not a guarantee of the historical Molab setup. Do not substitute it silently.
- `results/aggregated`: non-patient-level numeric results; `paper/`: figures/tables.
- `tests/`: synthetic functional checks, not full experiments; `.github/`: CI recipe.

## Evidence limits

Training: 1,059 patient groups / 138,399 slices. Evaluation: 37 previously inspected
validation patients / 4,761 slices, also used for encoder checkpoint selection.
The same 37 registered test patients were evaluated with earlier models; they are
not fresh confirmation. No expanded-model test results are reported.
Historical expanded-run encoder weights are unavailable. Separate preserved reruns
retain their own encoders and SAE states; they do not replace the historical vectors.
See `docs/PRESERVED_REPRODUCTION.md`. Saved embeddings and SAE states support downstream
analysis, not original image-to-embedding inference or full model reproduction.
The local `dai_analysis_reproduction_20261029` package additionally contains
checksummed analysis inputs, training-only feature-selection code, and expected
summaries. Its verification and full 18-dictionary selection rerun passed.
No verified external-generalization or clinical claim is made.
SAE seeds are nested within encoder; patient intervals condition on fixed models.
Do not treat slices, dictionary units, repeated baselines or matching pairs as
independent patient samples. Supervised is an original-run reference, not a new
matched F1/AUROC evaluation in notebook 25.

## GitHub upload

Extract this directory, review `PUBLICATION_NOTES.md`, then create a GitHub repository
and use the commands in `GITHUB_PUSH.md`. No remote repository is created or pushed
by this ZIP. A GitHub Actions recipe is supplied, but it has not been run on GitHub.
No project license is assumed; see `LICENSE_NOTICE.md`.
