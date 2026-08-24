"""
fix_metadata_paths_v2.py — file_path ko filename + split se reconstruct karta hai
(string-replace ke bajaye) — isliye chahe purana path relative ho ya absolute,
kaam karega.
"""

import os
import pandas as pd

METADATA_CSV = r"C:\Users\Raaghav\Desktop\coding\research\brain-mri-ncl-simclr\data\processed_flair\metadata.csv"
DATA_ROOT = r"C:\Users\Raaghav\Desktop\coding\research\brain-mri-ncl-simclr\data\processed_flair"

df = pd.read_csv(METADATA_CSV)

before_sample = df["file_path"].iloc[0]

def rebuild_path(row):
    fname = os.path.basename(row["file_path"])  # sirf filename nikaalo, purana folder ignore
    return os.path.join(DATA_ROOT, row["split"], fname)

df["file_path"] = df.apply(rebuild_path, axis=1)
after_sample = df["file_path"].iloc[0]

df.to_csv(METADATA_CSV, index=False)

print(f"Before: {before_sample}")
print(f"After:  {after_sample}")
print(f"\n{len(df)} rows updated.")

# Sanity check — pehli 5 aur random kuch files actually exist karti hain kya
missing = 0
for p in df["file_path"]:
    if not os.path.exists(p):
        missing += 1

print(f"\nTotal files: {len(df)} | Missing: {missing}")
if missing == 0:
    print("Sab files sahi jagah mil gayin. Ready to train.")
else:
    print(f"WARNING: {missing} files nahi milin — check karo ki train/val/test folders "
          f"sahi se {DATA_ROOT} ke andar hain.")
