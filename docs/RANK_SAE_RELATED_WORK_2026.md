# Related work for the rank-control experiment

Focused primary-source review checked **26 October 2026**. This is not an exhaustive
systematic review. Recent arXiv preprints are distinguished from established
conference papers; an author's abstract claim is not treated as independent
verification. None of these papers validates this project's MRI results.

## Direct foundations

**Wang et al. — Non-negative Contrastive Learning (ICLR 2024).** The paper develops
nonnegative contrastive representations and reports benefits for feature analysis
and transfer. Appendix G.1 places the nonnegative activation at the projector's
output: feature interpretation uses projector outputs, while downstream probes
use backbone features. [Paper and Appendix G.1](https://arxiv.org/html/2403.12459v3)

**Implication for us:** NCL is prior work, not our invention. Our SAE inputs are
192-dimensional backbone CLS features before the nonnegative projector, so we
study an indirect effect of the training constraint. Projector-feature guarantees
must not be presented as guarantees for these SAE decompositions.

**Jing et al. — Understanding Dimensional Collapse in Contrastive Self-supervised
Learning (ICLR 2022).** Contrastive representations can occupy a smaller subspace
without becoming a single constant vector. Negative pairs do not automatically
exclude dimensional collapse. [Paper](https://arxiv.org/abs/2110.09348)

**Implication for us:** Nonconstant predictions and high SAE reconstruction cannot
by themselves dismiss the low-rank alternative. Our train-only PCA control is
motivated by this concern, but is not a direct replication of their training method.

## Geometry, whitening and rank controls

**Saraswatula and Klindt — Data Whitening Improves Sparse Autoencoder Learning
(2025 preprint; arXiv metadata reports AAAI 2026 XAI4Science Workshop acceptance).**
They study PCA whitening of SAE inputs and report improvements in interpretability
evaluations, including cases with slightly worse reconstruction.
[Paper](https://arxiv.org/abs/2511.13981)

**Implication for us:** Input covariance can influence the observed SAE comparison.
Our retained-component whitening and recoloring to an NCL-derived spectrum is
related, but is not ordinary full whitening and does not reproduce their study.
It tests a specified transformation, not a universal covariance-independent effect.

**Delafrooz Noroozi, Puranik and Madhow — Shaped SAEs: Expanding the Toolkit for
Mechanistic Interpretability (preprint, 19 October 2026).** They examine activation
shaping, including rank-reducing projection informed by output Jacobians, and
introduce a matched-filter SAE. Their reported gains depend on settings, and
projection-related downstream losses are considered separately.
[Paper](https://arxiv.org/abs/2610.27375)

**Implication for us:** Rank reduction is not automatically harmful or interpretable;
which directions remain matters. Our covariance PCA uses neither output Jacobians
nor their dictionary architecture. Retained information, probe accuracy and
original-coordinate fidelity must accompany the selectivity comparison. This is
recent related work, not evidence that our pending control succeeds.

## Dead latents and seed dependence

**Gao et al. — Scaling and evaluating sparse autoencoders (2024).** They examine
Top-K SAEs, reconstruction/sparsity and downstream evaluations. Appendix A.2
introduces AuxK, which uses inactive latents to approximate reconstruction residuals;
their ablations also examine initialization.
[Paper, including AuxK](https://arxiv.org/html/2406.04093v1)

**Implication for us:** The count of used dictionary units depends on training and
optimization. Fewer alive features do not identify a smaller intrinsic concept set.
Our current rank control is baseline Top-K; an actual matched AuxK experiment is
still needed before claiming robustness to dead-latent mitigation.

**Paulo and Belrose — Sparse Autoencoders Trained on the Same Data Learn Different
Features (2025 arXiv preprint).** They report differences between dictionaries
trained with different initialization seeds, with Top-K showing notable dependence
in their language-model experiments. [Paper](https://arxiv.org/abs/2501.16615)

**Implication for us:** Keep encoder seeds and SAE seeds separate and retain
unfavorable runs. Similar mean purity across seeds does not show that the same
individual features recur. Feature correspondence requires a separate analysis.
The reviewer-mentioned ICLR 2026 venue label was not established by the arXiv
record inspected here; the bibliography uses the verified preprint metadata.

**Brzozowski and Chung — Aligned Training: A Parameter-Free Method to Improve
Feature Quality and Stability of Sparse Autoencoders (2026 preprint).** The authors
report improvements in dead features, reconstruction and stability through an
encoder–decoder alignment constraint.
[Paper](https://arxiv.org/abs/2605.18629)

**Implication for us:** SAE training geometry is another possible source of changes
in utilization and stability. This motivates an optimization control; it does not
show that alignment or AuxK will produce the same effect in our MRI representations.

## Evaluating interpretability and newer NCL variants

**Karvonen et al. — SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in
Language Model Interpretability (2025).** They evaluate several types of SAE
quality and find that proxy improvements need not translate to practical utility.
[Paper](https://arxiv.org/abs/2503.09532)

**Implication for us:** Reconstruction, used-unit counts and dominant-label purity
are separate measurements, not a single validated interpretability score. Our
purity/accuracy/spatial trade-offs should remain visible instead of selecting only
the metric favorable to NCL. We are not claiming to run the SAEBench benchmark.

**Cui, Zhang and Hu — Bayesian Gated Non-Negative Contrastive Learning (2026;
arXiv metadata reports ICML 2026 acceptance).** This work adds probabilistic gating
to address common/background-feature conflicts and reports benefits on ImageNet-100.
[Paper](https://arxiv.org/abs/2605.28441)

**Implication for us:** Current NCL research already extends beyond a terminal
nonnegative activation. Our method does not implement or benchmark this gated
variant; its reported semantic consistency or accuracy must not be attributed to
our MRI condition. It is relevant context for background/anatomical confounds.

## Literature-supported framing for this project

**The papers establish:** contrastive dimensional collapse is possible; covariance
and activation shaping can affect SAE outcomes; dictionary training and random
initialization affect the recovered features; proxy scores and utility can diverge.
These results arise in their own architectures and datasets.

**Our measured observation:** in the completed MRI validation runs, NCL uses fewer
SAE units and has higher dominant-slice-label purity, alongside lower backbone
effective rank than SimCLR and lower, less stable probe accuracy. This observation
comes from our archives, not the cited papers.

**Our inference to test:** part of the selectivity gap may reflect differences in
dimensionality/covariance rather than a rank-independent improvement in concepts.
Train-spectrum matching can test robustness under the registered transform. It
does not equate information content, basis orientation or all training dynamics.

If the gap shrinks, report that it is sensitive to this control. If it persists,
report robustness under this control. Neither result by itself proves causal
disentanglement, clinically meaningful tumour concepts or external generalization.

## Draft related-work paragraph

Nonnegative contrastive learning motivates a structural approach to representation
analysis, but projector-space interpretation and backbone-space transfer must be
distinguished (Wang et al., 2024). Contrastive learning can exhibit dimensional
collapse without constant outputs (Jing et al., 2022). More recent studies show
that whitening and activation shaping can change SAE evaluation outcomes
(Saraswatula and Klindt, 2025; Delafrooz Noroozi et al., 2026). We therefore examine
whether the observed backbone-SAE selectivity differences persist under train-only
rank and covariance-spectrum controls. Following work on dead latents, seed
dependence and multidimensional evaluation (Gao et al., 2024; Paulo and Belrose,
2025; Karvonen et al., 2025), we report dictionary utilization, reconstruction,
label selectivity and predictive utility separately. These controls constitute
an exploratory MRI study rather than a general guarantee of interpretability.

The innovation should not be described as inventing NCL, SAEs, PCA whitening or
rank controls. A defensible contribution is the controlled medical-imaging
evaluation and the documented relationship between training constraints,
representation geometry and SAE-based measurements.

## Reading and verification scope

Bibliographic metadata and abstracts were checked on the linked primary arXiv
records. Targeted full-text sections were checked for NCL's evaluation spaces,
Gao's AuxK mechanism and the shaping paper's projection/utility limitations.
The summaries do not claim an exhaustive reproduction of each paper's proofs or
experiments. Newer preprints should be described as preprints unless a proceedings
record is independently verified. Citation keys are in `rank_sae_related_work.bib`.
