# Preserved rerun verification

Six newly trained contrastive encoders/projectors and 18 native TopK SAE states
are retained with configurations, histories, train/validation embeddings, probes
and SHA256 fingerprints. All six archives passed CRC and fingerprint checks.
All 12 embedding arrays passed metadata, shape and finiteness checks; all 18 SAE
states loaded strictly and produced finite inference with the registered TopK bound.
Final encoders passed inference against saved synthetic reference tensors.
Real image-to-embedding replay also passed for three fixed validation slices per
encoder using the authorized preprocessed dataset. No test images were accessed.

Synthetic reference checks use rtol=1e-4, atol=1e-5. Real image replay across GPU/CPU
uses rtol=1e-3, atol=5e-4, with maximum errors reported individually. The initial
atol=1e-4 trial missed a near-zero SimCLR projection by 1.33e-4; the final report
uses the explicitly recorded numerical tolerance, not bitwise equivalence.
No full-cohort image replay, new full training, or full probe refitting was performed
as part of artifact verification. Environment versions are recorded.

These are preservation reruns, not recovered states of the historical encoders.
The original result tables are unchanged and their original encoders remain
unavailable. Historical downstream analysis has a separate archive. Model preservation
does not establish convergence or remove previous validation/test exposure.

Probe solver ceilings vary across archived runtimes. The three local SAE runs used
a ceiling of 20,000 iterations; the NCL seed-52 protocol records 10,000, and some
archived runtimes impose a 10,000 minimum internally despite a 2,000 nominal setting.
C grids, folds and objectives remain recorded per run. Do not treat the set as a new
uniform, confirmatory comparison or conceal execution overrides.

The local reproduction artifact is not publicly redistributed here. It contains
BraTS-derived embeddings/labels and model states and requires authorized access.
Code-only packaging excludes those assets and personal dataset repository defaults.
