"""
BraTS 2020 Preprocessing Script
--------------------------------
FLAIR modality se 2D axial slices extract karta hai, ground-truth tumor
sub-region labels ke saath, patient-level train/val/test split ke saath.

USAGE:
    1. Neeche CONFIG section mein DATA_ROOT apne dataset ke actual path pe set karo.
    2. `python preprocess_brats.py` chalao.
    3. Output: OUTPUT_DIR mein .npy slice files + a metadata CSV
       (patient_id, split, slice_idx, dominant_label, file_path)

Kaggle dataset (awsaf49/brats20-dataset-training-validation) ka typical structure:
    DATA_ROOT/
      BraTS2020_TrainingData/
        MICCAI_BraTS2020_TrainingData/
          BraTS20_Training_001/
            BraTS20_Training_001_flair.nii(.gz)
            BraTS20_Training_001_seg.nii(.gz)
            BraTS20_Training_001_t1.nii(.gz)
            ...
Agar tumhara structure alag hai, neeche find_patient_dirs() function adjust karo.
"""

import os
import glob
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import csv

# ==================== CONFIG ====================
DATA_ROOT = r"research\brain-mri-ncl-simclr\dataset\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"   # <-- YE BADALO apne actual path se
OUTPUT_DIR = r"research\brain-mri-ncl-simclr"   # <-- YE BHI BADALO

IMG_SIZE = 128                  # slices ko is size pe resize karenge (128x128)
MIN_BRAIN_PIXEL_FRACTION = 0.02  # slice mein kam se kam itna % brain tissue hona chahiye
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.8, 0.1, 0.1
RANDOM_SEED = 42

# BraTS label mapping: 0=background, 1=NCR/NET, 2=edema, 4=enhancing tumor
LABEL_NAMES = {0: "non_tumor", 1: "necrotic_core", 2: "edema", 4: "enhancing_tumor"}
# ==================================================


def find_patient_dirs(data_root):
    """Patient folders dhoondo (jinme flair aur seg files hon)."""
    candidates = glob.glob(os.path.join(data_root, "**", "*_flair.nii*"), recursive=True)
    patient_dirs = sorted(set(os.path.dirname(c) for c in candidates))
    if not patient_dirs:
        raise FileNotFoundError(
            f"Koi *_flair.nii(.gz) file nahi mili {data_root} ke andar. "
            "DATA_ROOT path check karo, ya folder structure alag hai toh "
            "find_patient_dirs() adjust karo."
        )
    return patient_dirs


def load_patient_volumes(patient_dir):
    """Ek patient ke FLAIR aur segmentation volume load karo."""
    flair_path = glob.glob(os.path.join(patient_dir, "*_flair.nii*"))[0]
    seg_candidates = glob.glob(os.path.join(patient_dir, "*_seg.nii*"))
    seg_path = seg_candidates[0] if seg_candidates else None

    flair = nib.load(flair_path).get_fdata()
    seg = nib.load(seg_path).get_fdata() if seg_path else np.zeros_like(flair)
    return flair, seg


def normalize_slice(slice_2d):
    """Non-zero voxels pe z-score normalization, phir 0-1 scale."""
    nonzero = slice_2d[slice_2d > 0]
    if nonzero.size == 0:
        return slice_2d.astype(np.float32)
    mean, std = nonzero.mean(), nonzero.std() + 1e-8
    normed = (slice_2d - mean) / std
    normed = np.clip(normed, -5, 5)
    normed = (normed - normed.min()) / (normed.max() - normed.min() + 1e-8)
    return normed.astype(np.float32)


def dominant_label(seg_slice):
    """Slice ka dominant tumor sub-region label (largest pixel-area wala)."""
    labels, counts = np.unique(seg_slice, return_counts=True)
    # background (0) ko ignore karo jab tak koi aur label na ho
    nonzero_mask = labels != 0
    if nonzero_mask.sum() == 0:
        return 0  # pure non-tumor slice
    labels_nz, counts_nz = labels[nonzero_mask], counts[nonzero_mask]
    return int(labels_nz[np.argmax(counts_nz)])


def process_patient(patient_dir, out_dir, patient_id):
    """Ek patient ke saare valid axial slices extract + save karo."""
    flair, seg = load_patient_volumes(patient_dir)
    n_slices = flair.shape[2]
    records = []

    for z in range(n_slices):
        flair_slice = flair[:, :, z]
        seg_slice = seg[:, :, z]

        brain_fraction = (flair_slice > 0).sum() / flair_slice.size
        if brain_fraction < MIN_BRAIN_PIXEL_FRACTION:
            continue  # mostly empty slice, skip

        norm_slice = normalize_slice(flair_slice)
        norm_slice = resize(norm_slice, (IMG_SIZE, IMG_SIZE), preserve_range=True, anti_aliasing=True)
        seg_resized = resize(seg_slice, (IMG_SIZE, IMG_SIZE), order=0, preserve_range=True,
                              anti_aliasing=False).astype(np.uint8)

        label = dominant_label(seg_resized)
        has_tumor = int(label != 0)

        fname = f"{patient_id}_slice{z:03d}.npy"
        out_path = os.path.join(out_dir, fname)
        np.save(out_path, {
            "image": norm_slice.astype(np.float32),
            "seg_mask": seg_resized,
        }, allow_pickle=True)

        records.append({
            "patient_id": patient_id,
            "slice_idx": z,
            "file_path": out_path,
            "dominant_label": label,
            "label_name": LABEL_NAMES.get(label, "unknown"),
            "has_tumor": has_tumor,
        })

    return records


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    patient_dirs = find_patient_dirs(DATA_ROOT)
    patient_ids = [os.path.basename(d) for d in patient_dirs]
    print(f"Total patients mile: {len(patient_ids)}")

    # ---- Patient-level split (slice leakage se bachne ke liye) ----
    train_ids, temp_ids = train_test_split(
        patient_ids, train_size=TRAIN_FRAC, random_state=RANDOM_SEED
    )
    val_frac_of_temp = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    val_ids, test_ids = train_test_split(
        temp_ids, train_size=val_frac_of_temp, random_state=RANDOM_SEED
    )
    split_map = {pid: "train" for pid in train_ids}
    split_map.update({pid: "val" for pid in val_ids})
    split_map.update({pid: "test" for pid in test_ids})

    print(f"Split -> train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

    all_records = []
    for i, patient_dir in enumerate(patient_dirs):
        patient_id = os.path.basename(patient_dir)
        split = split_map[patient_id]
        split_out_dir = os.path.join(OUTPUT_DIR, split)
        os.makedirs(split_out_dir, exist_ok=True)

        records = process_patient(patient_dir, split_out_dir, patient_id)
        for r in records:
            r["split"] = split
        all_records.extend(records)

        if (i + 1) % 20 == 0 or (i + 1) == len(patient_dirs):
            print(f"  [{i+1}/{len(patient_dirs)}] patients processed, "
                  f"{len(all_records)} slices so far")

    # ---- Metadata CSV save karo ----
    csv_path = os.path.join(OUTPUT_DIR, "metadata.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "patient_id", "slice_idx", "file_path", "dominant_label",
            "label_name", "has_tumor", "split"
        ])
        writer.writeheader()
        writer.writerows(all_records)

    # ---- Summary stats save karo ----
    summary = {
        "total_patients": len(patient_ids),
        "total_slices": len(all_records),
        "split_counts": {
            split: sum(1 for r in all_records if r["split"] == split)
            for split in ["train", "val", "test"]
        },
        "tumor_slice_fraction": sum(r["has_tumor"] for r in all_records) / max(len(all_records), 1),
        "img_size": IMG_SIZE,
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== DONE ===")
    print(json.dumps(summary, indent=2))
    print(f"\nMetadata CSV: {csv_path}")
    print(f"Processed slices: {OUTPUT_DIR}/train, /val, /test")


if __name__ == "__main__":
    main()
