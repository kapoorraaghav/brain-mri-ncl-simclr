"""Validate the six historical aggregate CSVs without changing them."""

import argparse
import csv
from itertools import combinations, product
import math
from pathlib import Path
import sys

METHODS = ("ncl", "simclr")
SEEDS = ("10", "42", "52")
SAE_CONDITIONS = ("auxk", "baseline")
PROBE_CONDITIONS = ("native", "pca_rank", "retained_95", "retained_99", "spectrum_matched")
ENCODER_KEYS = ("method", "encoder_seed", "condition")
STABILITY_METRICS = ("train_correlation_mean", "validation_correlation_mean", "validation_jaccard_mean")
SAE_GROUPS = set(product(METHODS, SEEDS, SAE_CONDITIONS))
SCHEMAS = {
    "feature_stability_pairs.csv": (
        ENCODER_KEYS + ("left_sae_seed", "right_sae_seed"), STABILITY_METRICS,
        {group + pair for group in SAE_GROUPS for pair in combinations(SEEDS, 2)}, 36,
    ),
    "feature_stability_per_encoder.csv": (ENCODER_KEYS, STABILITY_METRICS, SAE_GROUPS, 12),
    "stability_per_encoder.csv": (ENCODER_KEYS, STABILITY_METRICS, SAE_GROUPS, 12),
    "probe_all_30.csv": (
        ENCODER_KEYS, ("accuracy", "macro_f1", "balanced_accuracy", "auroc"),
        set(product(METHODS, SEEDS, PROBE_CONDITIONS)), 30,
    ),
    "probe_patient_intervals.csv": (
        ("comparison", "method", "condition", "metric"), ("point", "ci_low", "ci_high"), None, 150,
    ),
    "selectivity_patient_intervals.csv": (
        ("comparison", "method", "encoder_seed", "condition", "metric"),
        ("point", "ci_low", "ci_high"), None, 120,
    ),
}


def validate_table(path):
    """Check required columns, unique keys and finite primary metrics."""
    keys, metrics, expected, count = SCHEMAS[path.name]
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or not set(keys + metrics).issubset(headers):
            raise ValueError(f"{path.name}: invalid or missing columns")
        for line, row in enumerate(reader, 2):
            if None in row or None in row.values():
                raise ValueError(f"{path.name}:{line}: malformed row")
            key = tuple(row[column] for column in keys)
            if key in seen:
                raise ValueError(f"{path.name}:{line}: duplicate key {key}")
            seen.add(key)
            for column in metrics:
                try:
                    finite = math.isfinite(float(row[column]))
                except ValueError:
                    finite = False
                if not finite:
                    raise ValueError(f"{path.name}:{line}: invalid numeric value in {column}")
    if expected is not None and seen != expected:
        raise ValueError(f"{path.name}: unexpected method, seed or condition coverage")
    if len(seen) != count:
        raise ValueError(f"{path.name}: expected {count} rows, found {len(seen)}")
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("results/aggregated"))
    options = parser.parse_args()
    try:
        for name in SCHEMAS:
            rows = validate_table(options.directory / name)
            print(f"{name}: {rows} rows")
    except (OSError, ValueError, csv.Error) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
