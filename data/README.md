# Authorized inputs only

No data is distributed. Put your authorized `rank_control_embeddings_v1.zip` here
for analysis. `configs/rank_control_embeddings_v1_receipt.json` pins its identity.
Notebook 23/patient analysis also needs the completed AuxK model ZIP named in
configs/auxk_completed_models_receipt.json. Other stages do not need that model ZIP.

The embedding archive has input_manifest.json, split_manifest.csv and per-run
train/val NPZ matrices with sample identities. Metadata containing a test split
does not mean test embeddings are present. Only train/val matrices are supported.

Historical training requires the custom indexed preprocessed PKL format used by
the supplied training notebooks, not an arbitrary pickle. Only load trusted,
authorized PKLs. Preprocessing 13b needs original overlap mapping and older
manifest/metadata, all absent here. Path overrides must be reviewed before running.
Check applicable BraTS release-level data terms/citations before sharing any
derived dataset. This code ZIP does not redistribute it.
