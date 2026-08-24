# NCL + SAE for Interpretable Brain MRI Representations

Research project investigating whether Non-Negative Contrastive Learning (NCL) produces more interpretable representations than standard contrastive learning (SimCLR) on brain MRI, using Sparse Autoencoders (SAE) to decode and quantify feature interpretability.

**Author:** Raaghav Kapoor, B.Tech IT, MAIT Delhi
**Target venue:** ML/interpretability workshop (open, no fixed deadline)
**Dataset:** BraTS 2020 (FLAIR modality)

---

## Problem

Deep learning models detect brain tumors accurately but can't explain what they see — internal representations are entangled (a single dimension can encode tumor boundary, skull shape, and noise simultaneously via subtraction). This blocks clinical trust.

## Approach

1. **NCL** — a one-line change to SimCLR (add ReLU to the final projection layer) that bans negative values, removing the subtraction shortcut that causes entangled features.
2. **SAE** — trained on frozen backbone representations after contrastive/supervised training, decomposes them into sparse, individually-inspectable features.
3. **Compare** Supervised, SimCLR, and NCL backbones on how monosemantic (interpretable) their SAE features are.

See `pipeline_diagram.svg` / `pipeline_diagram.png` for the full architecture (all three branches, backbone → projection head → contrastive loss, and separately backbone → frozen representation → SAE).

---

## Repository Structure

```
.
├── 01_data_preprocessing.ipynb     # BraTS2020 NIfTI -> 2D FLAIR slices, patient-level split, labels
├── train_supervised.ipynb          # Supervised ViT-Tiny baseline (not shown here, same pattern as train_ncl)
├── train_simclr.ipynb              # SimCLR contrastive pretraining
├── train_ncl.ipynb                 # NCL contrastive pretraining (ReLU-constrained projector)
├── sae_supervised.ipynb            # SAE trained on frozen supervised representations
├── sae_simclr.ipynb                # SAE trained on frozen SimCLR representations
├── sae_ncl.ipynb                   # SAE trained on frozen NCL representations
├── results_summary.ipynb           # Loads final_results_table.json, builds comparison charts
├── generate_results_json.py        # Auto-generates results_template.json from training/eval outputs
├── results_template.json           # Consolidated results (metrics, backbone status, SAE stats)
├── pipeline_diagram.svg / .png     # Architecture diagram (editable SVG + rendered PNG)
├── NCL_SAE_Brain_MRI_Draft.docx    # Paper draft (Word)
├── NCL_SAE_Brain_MRI_Draft.pdf     # Paper draft (PDF)
└── README.md                       # This file
```

---

## Pipeline

```
BraTS 2020 FLAIR slice
        │
        ├── Supervised: ViT-Tiny → Classification Head (cross-entropy)
        │
        ├── SimCLR: 2 augmented views → shared ViT-Tiny → Projection Head (unconstrained)
        │            → contrastive loss (training only)
        │
        └── NCL: 2 augmented views → shared ViT-Tiny → Projection Head (+ReLU, non-negative)
                     → contrastive loss (training only)

For all three:
    Backbone output (frozen after training) → Sparse Autoencoder (TopK)
        → top-10 activating MRI slices per feature → Monosemanticity Check (Purity / Entropy)
```

## Data

- **Dataset:** BraTS 2020, ~370 patients, FLAIR modality (attach via Kaggle: `awsaf49/brats20-dataset-training-validation` or similar).
- **Split:** patient-level 70/15/15 (train/val/test) — no slice leakage across splits.
- **Preprocessing:** 2D axial slice extraction, percentile normalization, resize to 128×128, empty-slice filtering.
- **Labels:** per-slice dominant tumor sub-region (necrotic core / edema / enhancing tumor / non-tumor) from BraTS segmentation masks — used later for the monosemanticity metric, not for NCL/SimCLR training itself.

## Training

- **Backbone:** ViT-Tiny (identical across all three models for a fair comparison)
- **Hardware:** free-tier GPU sufficient — Kaggle T4×2 or a local RTX 5060 8GB
- **Batch size:** 64 | **LR:** 3e-4 | **Temperature:** 0.5 | **Epochs:** 25 | **Seed:** 42
- **Projector:** `Linear(192→192) → BatchNorm → ReLU → Linear(192→128)` (+ final ReLU for NCL)
  - BatchNorm in the projector was a required fix — without it, NCL's non-negativity constraint made representation collapse (loss flatlining at `ln(2×batch_size−1)`, the mathematical signature of all embeddings mapping to one point) much more likely than for SimCLR.

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Monosemanticity purity** | Mean fraction of an SAE feature's top-10 activating slices sharing the same dominant BraTS label |
| **Monosemanticity entropy** | Entropy of labels within an SAE feature's top-10 set (lower = more monosemantic) |
| **Downstream probe accuracy** | Linear probe on frozen representations, tumor vs. non-tumor |
| **Spatial Dice (ablation)** | Overlap of high-activation patches with the tumor mask |

## Results (latest run)

| Metric | Supervised | SimCLR | NCL |
|---|---|---|---|
| Monosemanticity purity | 0.898 | 0.796 | 0.843 |
| Monosemanticity entropy | 0.205 | 0.439 | 0.329 |
| Downstream probe accuracy | 0.861 | 0.865 | **0.877** |
| Spatial Dice | 0.011 | 0.286 | 0.175 |
| Alive SAE features | 59 | 499 | 44 |

Full machine-readable results (including SAE loss, best feature index, training config) are in `results_template.json` — regenerate it any time with:

```bash
python generate_results_json.py
```

**Takeaway so far:** NCL beats SimCLR on both interpretability metrics (purity and entropy) while also getting the best downstream probe accuracy of the three — the interpretability gain doesn't come at an accuracy cost. Supervised still edges out NCL on raw monosemanticity, which is expected since it uses label information directly during training.

---

## Reproducing

1. Run `01_data_preprocessing.ipynb` (produces `slices/`, `manifest.csv`, `split.json`)
2. Run `train_supervised.ipynb`, `train_simclr.ipynb`, `train_ncl.ipynb` (each saves a checkpoint + `<mode>_history.json`)
3. Run `sae_supervised.ipynb`, `sae_simclr.ipynb`, `sae_ncl.ipynb` (each writes into `final_results_table.json`)
4. Run `python generate_results_json.py` to consolidate everything into `results_template.json`
5. Run `results_summary.ipynb` to generate comparison charts and the qualitative top-10 activation grid
6. Update `NCL_SAE_Brain_MRI_Draft.docx` Section 7 with the final numbers

## Known Limitations

- NCL only blocks the *subtraction* shortcut — weakly-correlated concepts can still share a dimension, so full disentanglement isn't guaranteed.
- No radiologist validation yet; monosemanticity is judged only by automated purity/entropy against BraTS labels.
- Single modality (FLAIR); T1/T2/DWI extension left for future work.
- Dataset size (~370 patients) is modest — generalization to other scanners/cohorts untested.

## References

- Chen et al. (2020), SimCLR — arXiv:2002.05709
- Wang et al. (2024), NCL — arXiv:2403.12459
- Bricken et al. (2023), Towards Monosemanticity — transformer-circuits.pub/2023/monosemantic-features
- Wang et al. (2024), Beyond Interpretability — arXiv:2410.21331
- Menze et al., BraTS 2020 Dataset — Kaggle / Synapse
