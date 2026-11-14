# Workflow map and input provenance

This map distinguishes historical analyses from preservation reruns and the
current feature-population audit. It describes the full planned release; only
the first batch is currently published. Planned paths are not yet runnable.

## Available in the first batch

1. **Preprocessing:** `notebooks/13b_brats2023_preprocessing_fixed.ipynb` documents
   the patient-grouped BraTS expansion. It requires authorized images, original
   overlap mappings and earlier split metadata that are not distributed. Review
   its configuration; it is not a one-click recipe from an arbitrary dataset.
2. **Historical training:** the nine `notebooks/historical_training/20_*` files
   archive seeds 10, 42 and 52 for NCL, SimCLR and supervised training. They
   require the trusted indexed PKL format. Their filenames do not imply that the
   historical encoder weights survived. Use a later preserved workflow for new
   runs requiring image-level inference reproduction.
3. **General CLI:** `scripts/revision_experiments.py` is an earlier training
   interface. Its adjacent JSON settings are historical configurations, not a
   guarantee of the final notebook setup. The expanded configuration includes
   earlier AuxK and patch settings; do not substitute it silently.
4. **Embedding stage wrapper:** `scripts/run_embedding_experiment.py` supports
   help and dry-run now. It combines the shared runtime with stage-specific
   extensions and registers a runtime checksum. Full execution additionally
   needs the input receipts and extensions assigned to the second batch.

```bash
python scripts/run_embedding_experiment.py --stage rank --dry-run
python scripts/run_embedding_experiment.py --stage probe --dry-run
```

These commands do not load patient data, train a model or verify reproduction.

## Planned preserved training and earlier controls: batch 2

- `notebooks/30_ncl_*` and `30_simclr_*` preserve separate encoder runs.
  `notebooks/31_restore_preserved_saes_molab.py` restores matching SAE analysis.
  Use authorized PKLs and preserved exports. These models are a separate family,
  not replacements for the missing historical encoder states.
- Notebook 21 runs native, rank and spectrum controls. Notebook 22 compares
  native baseline and an adapted AuxK intervention, not the full Gao recipe.
- Notebook 23 evaluates patient bootstrap and feature stability from completed
  baseline/AuxK states. It needs both the historical embedding bundle and the
  completed AuxK model archive.
- Notebook 24 uses training-fitted retained-variance PCA controls. Matching
  retained variance does not match dimensions or predictive information.
- Notebook 25 fits raw-embedding probes, not SAE-code probes. It requires the
  identity-checked embedding bundle; AuxK does not modify those embeddings.

The exact embedding and model identities are recorded in the receipt JSON files
assigned to batch 2. Do not invent replacement receipts for a different archive.

## Planned current feature-population audit: batches 2 and 3

The notebook 32 entry point and `scripts/feature_population_audit.py` use
historical embeddings, matching SAE states and private per-slice validation
covariates. The supervised SAE reference is fitted to archived supervised
vectors, not a newly trained supervised encoder. Batch 3 adds the conditional
null controls, patient-resampling scripts and saved aggregate summaries.

`work/reviewer_sae_code_probes.py`, also assigned to batch 3, fits predictive
probes on frozen SAE activations. It must not be confused with notebook 25's
raw-embedding evaluation. Current paper correlations are descriptive; archived
resampling diagnostics do not imply calibrated confidence intervals.

## Results and model-family boundaries

All current audits reuse the checkpoint-selection validation cohort. The
registered test split was exposed earlier and is not fresh confirmation. Models,
selection rules, patient units and input identities must accompany comparisons.
After changing registered settings or code, use a new output directory rather
than mixing results. Export complete weights/results before closing ephemeral
cloud runtimes. Earlier legacy checkpoints already on the remote are not verified
inputs for the expanded historical audit.

The historical synthesis is assigned to batch 4. The current manuscript is
assigned to `paper/current/` in batch 5 and supersedes that earlier claim wording.
