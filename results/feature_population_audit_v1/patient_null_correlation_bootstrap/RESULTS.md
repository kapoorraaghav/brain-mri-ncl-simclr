# Completed paired patient uncertainty analysis

All 27 historical dictionaries evaluated; no encoder or SAE retraining.
1,000 whole-patient draws, paired across every method and fitted state; 1,000
within-stratum label permutations per draw. Three SAE seeds are averaged within
encoder before the three encoder values are averaged. Patient draws are reused
from the budget bootstrap, not generated independently by method.

## Results

Values are original point estimates and empirical 2.5th/97.5th patient-bootstrap
percentiles. Excess means and their gap below are in percentage points; rho is
Spearman correlation. These are conditional resampling ranges, not calibrated
confidence guarantees for nonsmooth, support-dependent top-ten statistics.

| Method | Statistic | Point | Patient-percentile range |
|---|---|---:|---:|
| ncl | excess_mean | 4.521 | [2.266, 7.638] |
| ncl | raw_rho | 0.174 | [0.109, 0.227] |
| ncl | excess_rho | -0.018 | [-0.025, 0.124] |
| simclr | excess_mean | 4.850 | [2.929, 9.016] |
| simclr | raw_rho | 0.328 | [0.258, 0.383] |
| simclr | excess_rho | -0.012 | [-0.006, 0.119] |
| supervised | excess_mean | 4.701 | [3.170, 9.597] |
| supervised | raw_rho | 0.297 | [0.206, 0.325] |
| supervised | excess_rho | 0.023 | [-0.031, 0.093] |
| ncl_minus_simclr | excess_mean | -0.329 | [-3.182, 1.266] |

## Interpretation and scope

The -0.33-point conditional-excess gap spans zero: report that NCL's lead is not
retained, not that SimCLR wins. Raw purity/AUROC correlations remain positive
across the percentile ranges; the stratified-excess ranges span zero. This does
not establish independence or prove that purity is a utility proxy.

Duplicating whole patients duplicates their slices. Recomputed top-ten lists and
support can therefore shift bootstrap distributions, particularly the excess
correlations: SimCLR's original -0.012 point is slightly below its percentile
range [-0.006, 0.119]. We retain and disclose this discrepancy rather than moving
the point estimate or silently centering the range. These are sensitivity ranges;
we make no equivalence claim or calibrated significance claim from them.

The full dictionary and training-fitted score orientations remain fixed. The
all-eligible support rule is reapplied each draw, including units that become
scorable through repeated copies. This differs from the budget analysis, where
selected IDs stay fixed and missing-support bounds are retained. No fresh
patients, independent confirmation, encoder-seed sampling, or fitting/analysis-
selection uncertainty are supplied by this bootstrap. All 1,000 draws yield
finite aggregate statistics.

Using 500 rather than 1,000 inner permutations changes the gap's lower/upper
percentile endpoints by 0.0072/0.0182 percentage points; the largest per-draw gap
change is 0.0710 points. This checks finite-permutation sensitivity, not bootstrap
coverage. State hashes, source hashes, eligibility ranges, finite fractions and
inner-permutation sensitivity are retained in the machine-readable outputs.

## Verification and reproduction

`python scripts/bootstrap_null_correlations.py` replays the calculations;
`python scripts/bootstrap_null_correlations.py --verify` independently checks
all 27 original full-dictionary AUROCs against the archived sparse-score results,
original eligibility/purity, literal expanded top tens and sklearn AUROC toy
cases, all 1,000 draw totals and stratum label counts for three permutation
indices per draw. The regeneration cache under tmp/ is excluded from archives.
Notebook 32 displays results by default; its optional replay flag stays False.
