# Stage order and what is saved

1. Obtain authorized processed data/embeddings. Preserve patient-group splits and
   the original mapping/provenance; do not fabricate the missing older inputs.
2. Historical training notebooks 20 document the original nine runs. They embed
   their runners and need no src imports. Their misleading checkpointed filenames
   are retained for traceability: no encoder weights survive those runs. For new
   image-based evaluation, a separately registered encoder run must save final
   weights. No silent training persistence changes were made to this public copy.
3. Notebook 21: native/rank-8/spectrum controls. NCL PCA and matched spectra
   coincide; they are not independent confirmations.
4. Notebook 22: matched native baseline/AuxK, not the full Gao recipe.
5. Notebook 23: paired patient bootstrap and train-matched feature stability. Uses
   saved AuxK/baseline weights, no retraining. Requires 37 validation patients for
   the reported cohort; thresholds and counts are explicit in the protocol.
6. Notebook 24: >=95% and >=99% train-variance PCA controls. Dimensions differ;
   variance retention does not match predictive information.
7. Notebook 25: refit small raw-embedding probes and evaluate all five conditions.
   It is not SAE-code evaluation, and AuxK does not change the input embeddings.

Current stages 21/22/24 save one final SAE model per completed fit; there is no
incomplete-epoch checkpoint recovery. Historical notebooks 20 are different and
save no final encoder weights. Always export whole result ZIPs before closing
ephemeral runtimes. Probes save predictions/probabilities and parameters; analysis
exports retain bootstrap replicates and matches.

The supplied core src runtimes use appended extensions, not subclasses: the
stage runner concatenates base+extension and sets a runtime SHA256. This preserves
the original experiment implementation. Do not edit registered code/settings and
reuse an old output folder. A new implementation is a new protocol/output path.

The two JSON files beside revision_experiments.py are historical settings (including earlier AuxK=96 and patch experiments), NOT the final notebook-22 AuxK=192 plan. The current stage runner and notebooks define their own registered plans.
