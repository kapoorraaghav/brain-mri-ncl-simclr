"""
Generates results_template.json automatically from the actual source files
produced by training + SAE notebooks. Run this instead of hand-editing the
JSON.

Expected inputs (in RESULTS_ROOT):
    - final_results_table.json   <- written by sae_supervised.ipynb / sae_simclr.ipynb / sae_ncl.ipynb
    - supervised_history.json    <- written by train_*.ipynb (save_results), flat OR in RESULTS_ROOT/supervised/
    - simclr_history.json
    - ncl_history.json

Usage:
    python generate_results_json.py
"""
import json
import os
import time

RESULTS_ROOT = r"C:\Users\Raaghav\Desktop\coding\research\brain-mri-ncl-simclr\results"
OUTPUT_PATH = os.path.join(RESULTS_ROOT, "results_template.json")
MODES = ["supervised", "simclr", "ncl"]

METRIC_MAP = {
    "monosemanticity_purity": (
        "mean_purity",
        "Mean purity over top-K=10 activating slices per alive SAE feature, using BraTS sub-region labels",
    ),
    "monosemanticity_entropy": (
        "mean_entropy",
        "Mean entropy over top-K=10 activating slices per alive SAE feature (lower = more monosemantic)",
    ),
    "downstream_probe_accuracy": (
        "downstream_probe_accuracy",
        "Linear probe accuracy on frozen representations, tumor vs non-tumor classification",
    ),
    "spatial_dice_alignment": (
        "spatial_dice",
        "Optional/ablation: Dice overlap of high-activation patches vs tumor mask",
    ),
}


def find_history_path(results_root, mode):
    """History file can be at RESULTS_ROOT/<mode>_history.json (flat) or
    RESULTS_ROOT/<mode>/<mode>_history.json (subfolder) -- check both."""
    candidates = [
        os.path.join(results_root, f"{mode}_history.json"),
        os.path.join(results_root, mode, f"{mode}_history.json"),
        os.path.join(results_root, mode, "results", f"{mode}_history.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_json(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def build_backbone_entry(mode, history):
    if history is None:
        base = {"status": "not_started", "checkpoint_path": None}
        if mode == "supervised":
            base["final_train_loss"] = None
            base["best_val_acc"] = None
        else:
            base["final_contrastive_loss"] = None
            base["best_loss"] = None
            if mode == "ncl":
                base["collapsed"] = None
        return base

    entry = {"status": "done", "checkpoint_path": None}
    if mode == "supervised":
        entry["final_train_loss"] = history["train_loss"][-1] if history.get("train_loss") else None
        entry["best_val_acc"] = max(history["val_acc"]) if history.get("val_acc") else None
    else:
        entry["final_contrastive_loss"] = history["train_loss"][-1] if history.get("train_loss") else None
        entry["best_loss"] = history.get("best_loss")
        if mode == "ncl":
            entry["collapsed"] = False  # set True manually if a run actually collapsed
    return entry


def build_sae_section(final_results):
    if not final_results:
        return {
            "dictionary_size": None,
            "sparsity_k": None,
            "trained_on": MODES,
            "status": "not_started",
            "per_backbone": {},
        }

    per_backbone = {}
    for mode in MODES:
        r = final_results.get(mode)
        if r:
            per_backbone[mode] = {
                "sae_final_loss": r.get("sae_final_loss"),
                "n_alive_features": r.get("n_alive_features"),
                "best_feature_idx": r.get("best_feature_idx"),
                "best_feature_purity": r.get("best_feature_purity"),
            }

    return {
        "dictionary_size": None,  # fill manually if not logged in final_results_table.json
        "sparsity_k": None,       # fill manually if not logged in final_results_table.json
        "trained_on": MODES,
        "status": "done" if len(per_backbone) == len(MODES) else "in_progress",
        "per_backbone": per_backbone,
    }


def build_evaluation_metrics(final_results):
    metrics = {}
    for out_key, (src_key, description) in METRIC_MAP.items():
        row = {"description": description}
        for mode in MODES:
            r = final_results.get(mode) if final_results else None
            row[mode] = r.get(src_key) if r else None
        metrics[out_key] = row
    return metrics


def build_hypothesis_summary(final_results):
    if not final_results or not all(m in final_results for m in MODES):
        return {}

    sup, sim, ncl = final_results["supervised"], final_results["simclr"], final_results["ncl"]
    best_probe_mode = max(MODES, key=lambda m: final_results[m]["downstream_probe_accuracy"])

    return {
        "ncl_vs_simclr_purity": "NCL better" if ncl["mean_purity"] > sim["mean_purity"] else "SimCLR better",
        "ncl_vs_simclr_entropy": "NCL better (lower)" if ncl["mean_entropy"] < sim["mean_entropy"] else "SimCLR better",
        "ncl_vs_supervised_purity": "NCL better" if ncl["mean_purity"] > sup["mean_purity"] else "Supervised better",
        "ncl_vs_supervised_entropy": "NCL better (lower)" if ncl["mean_entropy"] < sup["mean_entropy"] else "Supervised better",
        "best_downstream_probe_accuracy_mode": best_probe_mode,
        "alive_features": {m: final_results[m]["n_alive_features"] for m in MODES},
    }


def main():
    final_results = load_json(os.path.join(RESULTS_ROOT, "final_results_table.json"))
    histories = {mode: load_json(find_history_path(RESULTS_ROOT, mode)) for mode in MODES}

    data = {
        "_comment": "Auto-generated by generate_results_json.py. Do not hand-edit -- rerun the script instead after new training/eval runs.",
        "project": "NCL + SAE for Interpretable Brain MRI Representations",
        "dataset": "BraTS 2020 (FLAIR modality)",
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_config": {
            "img_size": 128,
            "batch_size": 64,
            "learning_rate": 3e-4,
            "temperature": 0.5,
            "epochs": 25,
            "seed": 42,
            "backbone": "ViT-Tiny",
            "projector": "Linear(192->192) + BatchNorm + ReLU + Linear(192->128) [+ ReLU if NCL]",
        },
        "backbones": {mode: build_backbone_entry(mode, histories[mode]) for mode in MODES},
        "sae": build_sae_section(final_results),
        "evaluation_metrics": build_evaluation_metrics(final_results),
        "qualitative": {
            "top10_activation_grid_figure_path": os.path.join(RESULTS_ROOT, "figures_top10_combined.png")
            if os.path.exists(os.path.join(RESULTS_ROOT, "figures_top10_combined.png")) else None,
            "notes": "",
        },
        "hypothesis_check_summary": build_hypothesis_summary(final_results),
        "paper_section_mapping": {
            "section_4_experimental_setup": "training_config + backbones",
            "section_5_evaluation_metrics": "evaluation_metrics",
            "section_7_results_table": "evaluation_metrics + qualitative",
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved: {OUTPUT_PATH}")
    missing = [m for m in MODES if histories[m] is None]
    if missing:
        print(f"NOTE: backbone training history missing for: {missing} -- final_train_loss/best_loss left as null")
    if final_results is None:
        print("NOTE: final_results_table.json not found -- SAE section + evaluation_metrics left as null")


if __name__ == "__main__":
    main()