"""Check the public aggregate validator using synthetic probe rows."""

import csv
from itertools import product
from pathlib import Path
import tempfile
import unittest

from scripts.validate_public_aggregates import (
    METHODS, PROBE_CONDITIONS, SEEDS, validate_table,
)


class PublicAggregateValidationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "probe_all_30.csv"
        self.rows = [(*key, "0.7", "0.6", "0.75", "0.8")
                     for key in product(METHODS, SEEDS, PROBE_CONDITIONS)]

    def write_rows(self):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("method", "encoder_seed", "condition", "accuracy",
                             "macro_f1", "balanced_accuracy", "auroc"))
            writer.writerows(self.rows)

    def test_complete_method_seed_condition_grid(self):
        self.write_rows()
        self.assertEqual(validate_table(self.path), 30)

    def test_missing_seed(self):
        self.rows = [row for row in self.rows if row[1] != "52"]
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_table(self.path)

    def test_duplicate_key(self):
        self.rows.append(self.rows[0])
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            validate_table(self.path)

    def test_nonfinite_and_nonnumeric_metrics(self):
        for value in ("nan", "inf", "missing"):
            with self.subTest(value=value):
                self.rows[0] = (*self.rows[0][:-1], value)
                self.write_rows()
                with self.assertRaisesRegex(ValueError, "invalid numeric"):
                    validate_table(self.path)
