"""Check that published notebooks contain no saved execution state."""

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PublicationHygieneTests(unittest.TestCase):
    def test_notebook_outputs_and_execution_counts_are_cleared(self):
        notebooks = sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb"))
        self.assertTrue(notebooks)
        for path in notebooks:
            document = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(document["cells"]):
                if cell["cell_type"] == "code":
                    with self.subTest(notebook=path.name, cell=index):
                        self.assertFalse(cell.get("outputs"))
                        self.assertIsNone(cell.get("execution_count"))
