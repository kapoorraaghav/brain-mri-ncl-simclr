# Feature-population audit: public code and aggregate outputs

This public release contains audit code, notebook 32 and aggregate summaries.
It omits private embeddings, per-slice patient covariates, activation arrays,
per-feature outputs and model-state archives. References to the private full
artifact do not mean those inputs are distributed in this repository.

The current analyses use historical frozen states, not the separately preserved
encoder reruns. Three primary input identities are registered in
`results/feature_population_audit_v1/audit_protocol.json`. The embedding hash also
agrees with `configs/rank_control_embeddings_v1_receipt.json`. An AuxK or preserved
rerun archive is not a substitute for the registered historical SAE archive.

## Read-only input preflight

```bash
python scripts/check_audit_inputs.py --dry-run
python scripts/check_audit_inputs.py
```

The dry-run reads the published protocol and checks input path presence, but
never hashes the private inputs. It reports fingerprint checks as deferred.
The normal mode streams SHA256 for the three primary inputs. Neither mode loads
models, unpickles arrays, extracts archives, produces predictions or writes outputs.
Only Python's standard library is needed.

Exit codes: 0 means the checks for the selected mode passed; 2 means an input
is absent or a fingerprint mismatches; 1 means an invalid protocol or unreadable
input. A successful dry-run is not input-identity verification. Matching primary
hashes does not verify payload validity, every downstream artifact or reproducible
inference. Model fitting and predictions are not performed by this helper.

Defaults match the existing audit script's paths. For an authorized local copy
stored elsewhere, override the paths explicitly:

```bash
python scripts/check_audit_inputs.py --root . --embeddings path/to/rank_control_embeddings_v1.zip --sae-models path/to/sae_models.zip --covariates path/to/validation_covariates.csv
```

`--protocol` selects a registration file relative to `--root` or an absolute path.
Use the original registration to reproduce the historical study. A changed input
with a newly registered hash is a new experiment, not evidence that the old
fingerprint matched. The helper rejects preservation-protocol versions.
Path overrides affect this helper only; they do not rewrite the audit script's
input constants or the notebook configuration.

## Reproduction entry points

Open `notebooks/32_feature_population_audit.ipynb` from the repository root.
It displays existing results by default. Numerical recomputation needs the
authorized private full-artifact inputs and its saved SAE states; enabling a
recompute flag without those artifacts cannot recover missing data or weights.
The supervised reference additionally needs its saved states/embedding cache or
the separately documented fitting inputs. The preflight checks three primary
historical inputs, not every supervised or downstream artifact.

```bash
python scripts/verify_feature_population_audit.py
python scripts/feature_population_audit.py --methods ncl simclr supervised --force-recompute
python work/report_feature_population_audit.py
```

The verification and report scripts also refer to private per-fit artifacts;
not every command can run against the public aggregates alone. Full verification
requires the authorized full artifact. Recomputing audits does not retrain the
historical encoders. A changed registered configuration needs a new output path.

See `results/feature_population_audit_v1/RESULTS_README.md` for the saved analysis
scope and aggregation. All outcomes reuse validation patients involved in
checkpoint selection. Null/correlation bootstrap files are archived conditional
sensitivity diagnostics, not calibrated confidence intervals. The current paper
reports point correlations only. No fresh patient confirmation or clinical
explanation validity is established. Scientific results were not recomputed as
part of this publication batch.
