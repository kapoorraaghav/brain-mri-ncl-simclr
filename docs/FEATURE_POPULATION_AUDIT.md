# Completed feature-population audit artifact

Open notebooks/32_feature_population_audit.ipynb in the extracted artifact root.
Dependencies: torch, numpy, pandas, scipy, scikit-learn, matplotlib and Jupyter.
The notebook displays existing results and verifies them by default. Enable its
RECOMPUTE_AUDITS flag to repeat all fixed-state calculations using archived inputs.
It does not retrain encoders. Nine new supervised SAE states are included; the
supervised fitter can use its saved embedding cache rather than original cloud ZIPs.

Command-line reproduction:

    python scripts/verify_feature_population_audit.py
    python scripts/feature_population_audit.py --methods ncl simclr supervised --force-recompute
    python work/report_feature_population_audit.py

See results/feature_population_audit_v1/RESULTS_README.md for analysis scope,
stratification and uncertainty. Historical preservation reruns are not the inputs
to these audits. Original encoder weights remain missing, and all new outcomes
reuse validation patients. Models, source embeddings, covariates and per-feature
outputs are included here for authorized local use, not public redistribution.
No raw MRI PKL or fresh test data are included. Keep this full archive private.
The separate CODE_AND_RESULTS ZIP excludes vectors, per-slice metadata and states.
The latter still needs authorized input artifacts to rerun numerical analyses.

THIS IS THE CODE/AGGREGATE-RESULTS VERSION: private input arrays and model states are omitted.
