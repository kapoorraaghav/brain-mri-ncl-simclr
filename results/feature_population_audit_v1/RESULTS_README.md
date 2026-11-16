# Feature-population audit results

`audit_protocol.json` freezes analysis settings before this run. All results reuse
the historical 37-patient validation cohort. NCL/SimCLR use the archived baseline
SAEs; the supervised reference adds nine freshly fitted SAEs on saved supervised
encoder embeddings, using the historical baseline initialization/optimizer recipe.
No encoder was retrained and no test images or new patients were evaluated.

`fit_summary.csv`, `per_encoder.csv` and `summary.csv` report feature-level Spearman
correlations and global/stratified null excess. Orientations use training tumour
labels only. Within-feature utility is univariate binary AUROC, not causal importance,
ablation utility, or evidence about interactions. Feature-level p-values are not used.

Stratification explicitly preserves zero tumour area and positive-area quantile bins
crossed with axial-index bins. It conditions on segmentation-derived covariates;
this is not a deployable label-free adjustment or a causal confound correction.
`strata_definition.json` exposes cutpoints, cell sizes, patients and class composition.
Three/five-bin sensitivities accompany the primary four-bin reference. Both nulls
use 1,000 fixed-seed permutations; the original paper used 100 global permutations.

`budget_fits.csv` extends each dictionary's curve to its own eligible endpoint.
Equal-J paired plots restrict to budgets with all 3 SAE seeds in all 3 encoder seeds
for both NCL/SimCLR. Beyond that domain, there is no fully replicated equal-J method
comparison. Method-specific full-set endpoints are not matched-budget comparisons.
Bounds preserve frozen training-selected IDs and assign unscorable values [0,1].

Random subsets use 1,000 uniform random permutations without replacement from the
same candidate pool; subsets are nested as J increases. The CSV contains per-fit
reference means, 2.5/97.5% quantiles and descriptive usage percentiles. These are
conditional feature-selection references, not patient confidence intervals/p-values.

Means average SAE seeds within encoder first; SD is across three encoder means.
Faint Figure A lines are individual dictionaries; thick lines have full replication
only. Figure B now uses the paired patient-bootstrap/support envelope when its outputs exist; otherwise it shows encoder SD. C shading is deterministic support bounds, not patient uncertainty.
Figure D shading uses 2.5/97.5% quantiles after aggregating 1,000 random-draw means
over the nine dictionaries in each method. It is a conditional random-selection
reference for the macro-average, not patient/model-training uncertainty.
Sign changes are descriptive crossing locations, not estimated change points.

## Paired patient-budget bootstrap

2,000 common draws resample whole validation patients with replacement, retaining all slices and duplicate multiplicity. Models and original selected IDs stay fixed. Top-ten lists are recomputed. Figure B uses the 2.5th percentile of the gap lower bound and 97.5th percentile of its upper bound when missing support makes purity undefined. This pointwise conditional envelope is not encoder SD, a simultaneous band or guaranteed nominal coverage. Strict complete-case intervals require at least 95% fully scorable draws. Null and correlation patient intervals are reported separately under patient_null_correlation_bootstrap/. Outputs, fixed IDs and draw counts are under patient_budget_bootstrap/. Reproduce with `python scripts/bootstrap_feature_budget_patients.py` before rebuilding the figure.

## Paired patient uncertainty for Tables 2 and 3

The new bootstrap uses the first 1,000 common whole-patient draws from the budget
bootstrap, across all 27 fixed dictionaries. It recomputes top-ten lists, positive
support, tumour-presence AUROC and the conditional null (1,000 within-cell label
permutations per patient draw, paired across methods). Patient copies retain every
slice, and copies are distinct permutation positions. Training-fitted AUROC
orientations and original four-bin cutpoints stay fixed. Cell label composition
is recomputed. The full dictionary stays fixed; the all-eligible support rule is
reapplied per draw, including formerly ineligible units. This is intentionally a
different estimand from the fixed-ID budget bounds. No unscorable units are
silently replaced within a fixed-budget comparison.

Percentile intervals are conditional, pointwise and exploratory; they do not
include model-training uncertainty or repair validation reuse. Duplicated slices
can shift top-ten order statistics and eligibility, so bootstrap means need not
match original point estimates. Finite fractions, eligibility ranges, and the
500-versus-1,000 inner-permutation sensitivity are reported. See
patient_null_correlation_bootstrap/{protocol.json,intervals.csv,SUMMARY.json}.
Reproduce with `python scripts/bootstrap_null_correlations.py`; independently
check with `python scripts/bootstrap_null_correlations.py --verify`.
The approximately 4.9 GB temporary permutation cache is regenerable, outside the
results directory, and excluded from delivered archives.
