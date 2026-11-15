# Preserved reproduction code only

This code archive excludes all model states, embeddings, clinical labels and raw MRI. It includes six encoder-training notebooks, a restore/SAE notebook, inference and artifact verification, requirements and a path-free verification summary.

To verify existing states, obtain the authorized private artifact bundle separately (ARTIFACTS.json plus artifacts/*.zip) and run `python scripts/verify_preserved_reproduction.py --bundle /path/to/bundle`. Add `--dataset /path/to/authorized.pkl` for three-slice validation replay per encoder. Raw data are neither required for saved-vector checks nor distributed here.

See VERIFICATION_NOTES.md for scope and solver differences. These are new preservation reruns, not recovered original-paper models. Historical tables are unchanged. No dataset, model-state or embedding redistribution authorization is implied. Personal dataset download defaults have been replaced with placeholders.
