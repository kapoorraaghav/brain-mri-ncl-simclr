# Representation Geometry and Sparse Autoencoder Feature Selectivity in Brain MRI

## Final framing

This is a controlled empirical study of **selectivity, representation geometry,
dictionary utilization and predictive utility**. It is not a new NCL architecture,
a new SAE algorithm, a diagnostic system or evidence that NCL is universally better.
The main architectural comparison is NCL versus SimCLR. Supervised is a reference
from the original three-way experiment, with different supervision/augmentation.

All numerical claims below refer to the expanded-cohort validation experiments,
not the original submission's single-run dictionary counts of 44/499 or its old
probe/Dice numbers. Do not mix those earlier results into the final tables.

## Evidence and supported claims

| Claim | Supporting evidence | Required qualification |
|---|---|---|
| NCL yields higher average dominant-label selectivity than SimCLR in native baseline SAEs. | T1/T3: purity 83.99% vs 79.87%; patient-distinct purity 78.84% vs 73.37%. | Labels describe dominant slice regions; purity does not establish spatial localization or causal/clinical concepts. |
| Geometry contributes to the sensitivity of the comparison. | T3: purity advantage shrinks from 4.11 points natively to 0.93 at rank 8 and 2.04 with spectrum matching. | Matching affects retained information and optimization; it does not isolate a causal effect of rank or equate semantics. NCL rank-8/spectrum conditions duplicate the same reference transform. |
| Higher selectivity persists on average under comparable training-variance retention. | T3: NCL-minus-SimCLR purity is +3.69 points at 95% and +5.12 at 99%. | 95% favors NCL in 2/3 encoders, 99% in 3/3. Targets can overshoot; retention fractions are not identical and do not equate task information. |
| Predictive utility depends strongly on the compression constraint. | T2/T6: native macro-F1 84.80% NCL vs 87.88% SimCLR; rank 8 reverses the average ranking; 99% retention gives 75.88% vs 87.01%. | Rank 8 preserves about 98.45% NCL but 35.02% SimCLR validation variance; that reversal is not evidence of general superiority. |
| The tested AuxK intervention retains a NCL purity advantage but does not resolve NCL utilization. | T4: alive NCL 198 to 168, SimCLR 1309 to 1488; AuxK purity 83.28% vs 79.88%. | This is an adapted auxiliary-loss ablation, not Gao's full initialization/normalization recipe. Do not conclude that all mitigation fails or inactive units are unnecessary. |
| Higher selectivity is not accompanied by a demonstrated stability advantage. | T5: baseline matched validation correlation NCL 0.669 vs SimCLR 0.710; AuxK 0.669 vs 0.678. | Training-eligible matched features only, within a fixed encoder. Different coverage and redundant directions matter; correlation does not identify semantic equivalence. |

## Suggested abstract

Sparse autoencoder (SAE) feature selectivity can be confounded by representation
geometry and dictionary utilization. We examine this issue in brain-MRI FLAIR
representations learned with nonnegative contrastive learning (NCL) and SimCLR,
using three encoder seeds and three SAE seeds per encoder. Encoders use 1,059
training patient groups; all reported evaluations use 37 validation patients.
We separately measure dominant-label selectivity, reconstruction, utilization,
predictive performance and within-encoder feature stability, with train-only
rank, covariance-spectrum and retained-variance controls. Native NCL SAEs show
higher average label purity than SimCLR (83.99% versus 79.87%), but lower
raw-embedding probe macro-F1 (84.80% versus 87.88%). The selectivity advantage
attenuates under spectrum matching and persists on average at comparable retained
variance, while predictive utility favors SimCLR at the 95% and 99% retention
targets. An adapted AuxK intervention improves SimCLR utilization but does not
alleviate NCL's low utilization, and feature matching does not show an NCL
stability advantage. These exploratory validation results demonstrate a
selectivity–utility trade-off and motivate evaluating SAE explanations through
multiple measurements rather than purity or alive-feature counts alone.

## Draft results text

### Native representations and three-way reference

With D=1536 and K=24, NCL uses an average of 198 validation-alive units compared
with 1309 for SimCLR, while positive activity per slice remains approximately
24 for both. Thus lower dictionary utilization is not stronger per-input
sparsity. The supervised reference and all K=8 results are reported alongside
the primary contrastive comparison, rather than omitted. Native NCL purity is
83.99% versus 79.87% for SimCLR; the patient-distinct top-ten metric is 78.84%
versus 73.37%. Reference shuffle levels and eligibility coverage are reported in
T1 and the exact source tables. No single metric is interpreted as overall
interpretability. The original spatial measurements are a CLS-to-patch transfer
diagnostic and do not establish superior tumour localization.

### Geometry-sensitive selectivity

Equal rank does not equal equal retained information. Rank-8 projection preserves
on average 98.45% of NCL validation variance but only 35.02% of SimCLR variance.
The average purity gap decreases to 0.93 percentage points at rank 8 and is 2.04
points after training-spectrum matching. A different sensitivity control chooses
the minimal training PCA rank meeting 95% or 99% retention. At 99%, average
validation variance retention is 99.08% for NCL and 98.98% for SimCLR, with purity
84.81% versus 79.69%. This supports robustness under the specified retention
control, not a covariance-independent or causal concept advantage. At 95%, seed
52 slightly favors SimCLR; both registered targets and all seeds are retained.

### Predictive utility

Refitted NB25 probes use the same training-only standardization and fixed
logistic-regression settings across conditions. Native SimCLR exceeds NCL in
macro-F1 (87.88% versus 84.80%), balanced accuracy, tumour recall, AUROC and average
precision. The validation binary labels are approximately balanced (2373 negative,
2388 positive slices), so class imbalance alone does not explain this pattern.
Rank-8 compression reverses average macro-F1 (80.75% NCL versus 76.36% SimCLR),
but the severe difference in retained variance limits its interpretation. Under
99% retention, macro-F1 is 75.88% for NCL versus 87.01% for SimCLR. NCL's larger
seed variation and the lower retained-variance AUROC show that the trade-off is
not confined to one accuracy threshold. These are raw-embedding probes, not SAE-
code probes: the utility results must not be attributed to the SAE itself.

### AuxK, reproducibility and uncertainty

The registered AuxK ablation changes only the auxiliary objective. SimCLR's mean
validation-alive count rises to 1488, whereas NCL's falls to 168 and its training
inactivity increases. Auxiliary activations are recorded, but improvement in NCL
utilization is not observed. NCL remains more label-selective under this variant.
Feature matching is fitted on a patient-balanced training subset and evaluated
on validation activations. Mean matched correlations are 0.669 for NCL and 0.710
for SimCLR in baseline dictionaries, without a demonstrated NCL stability gain.
One thousand shared patient-cluster bootstrap draws give conditional intervals
for paired selectivity and probe differences (T6/F4). They do not include model-
training variation, correct for all exploratory comparisons or convert reused
validation data into independent test evidence.

## Methods details to retain

- Expanded BraTS FLAIR cohort: training 1059 patient groups/138399 slices,
  validation 37/4761, held-out test 37/4821. Patient IDs do not cross splits in
  the registered manifest. Report patient-group counts alongside slices; do not
  treat slices, dictionary units, SAE seeds or repeated baselines as independent patients.
- NCL changes a terminal activation at the contrastive projector. SAE/probe
  inputs are the 192-dimensional backbone CLS vectors **before** that projector;
  they are not guaranteed nonnegative. Original NCL projector-space theoretical
  statements cannot be directly transferred to these backbone SAE decompositions.
- Encoder seeds and SAE seeds are 10/42/52; 50 encoder epochs and 30 SAE epochs.
  LR=0.0005; dictionary=1536; main K=24, original sensitivity K=8. Full original
  configuration/source hashes are preserved in the audited comparison.
- Training-only mean/PCA/spectrum transforms. Refit raw-embedding probes use
  StandardScaler + LogisticRegression(C=1,max_iter=2000); no validation-based
  threshold selection or class reweighting. All 30 new probes avoided the
  iteration limit. Software/environment and coefficients are recorded.
- Mean ± SD means sample SD across three encoder means after averaging SAE seeds
  when applicable. Probe tables have one result per encoder, not nine. Stability
  pairs are averaged within encoder before across-encoder reporting.
- Bootstrap intervals are 95% empirical percentile whole-patient intervals with
  common draws, conditional on fixed trained models. Non-smooth top-ten and
  eligibility statistics can give unreliable coverage. Count ranges are
  resampling sensitivity, not confidence bounds on population concept counts.

## Claims to exclude

Do not claim overall NCL superiority; better diagnosis or clinical interpretability;
proof of disentanglement, causal tumour concepts or absence of dimensional
collapse; benign/useless dead units; superiority to supervised learning;
cross-encoder semantic feature identity; external or held-out generalization;
or a general guarantee from the AuxK/rank/variance controls. No new comparison
supports a general localization advantage. Do not present 180 SAE result records
as independent scientific replications or count duplicate baseline conditions as
new evidence.

## Unresolved limitations and provenance

The expanded-run encoder weights and compatible test embeddings were not saved.
Older local test caches have a different manifest. Thus the registered 37-patient
test set was not evaluated, and no external cohort with verified independence
supports this package. Earlier Warsaw exploratory outputs are not final independent-
cohort evidence and are deliberately excluded. Training uses many patient groups,
but the uncertainty calculations still concern only 37 inspected validation patients.

The current NB25 native probes differ slightly from the original three-way probe
outputs; retain the original values in S4 and use NB25 consistently for the
current NCL–SimCLR predictive comparison. Do not silently combine old supervised
accuracy with new F1/AUROC into one supposedly identical three-way probe run.
The numerical shifts need source/environment reconciliation before claiming exact
reproducibility. There is no full new supervised F1/AUROC evaluation here.

## Suggested paper layout

Use F1 as the main results figure. T2 and T3 contain the predictive and geometry
comparisons; merge/subset their columns for the eventual venue layout, preserving
all conditions in the appendix. Use T1 to establish the original supervised
reference and utilization. AuxK, stability, patient intervals, K=8, spatial
transfer and full per-seed details can be placed in the supplement with references
from the main text. A short paper should not display every asset at once.

The conclusion should read: **nonnegative contrastive training is associated
with higher SAE label selectivity in this setting, but selectivity, dictionary
utilization, feature stability and predictive utility do not improve together.**
The controls support a qualified empirical trade-off, not universal superiority.
