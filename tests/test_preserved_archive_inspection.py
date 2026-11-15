"""Synthetic archive tests. No MRI, PyTorch, network, extraction or GPU needed."""
from pathlib import Path
import importlib.util
import json
import stat
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/inspect_preserved_archive.py'
spec = importlib.util.spec_from_file_location('preserved_inventory', SCRIPT)
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


class ArchiveInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.archive = self.folder / 'run.zip'
        self.prefix = 'final/ncl/fixed/seed_10/'

    def tearDown(self):
        self.temporary.cleanup()

    def members(self, saes=False):
        names = list(inventory.ROOT_REQUIRED)
        names += [self.prefix + name for name in inventory.RUN_REQUIRED]
        if saes:
            names += [self.prefix + f'evaluation/native/baseline/sae_{seed}/model.pt'
                      for seed in inventory.SAE_SEEDS]
        return names

    def write_zip(self, names):
        with zipfile.ZipFile(self.archive, 'w') as archive:
            for name in names:
                # Deliberately invalid JSON and weight contents: only names may be inspected.
                archive.writestr(name, b'not a serialized model or JSON document')

    def test_expected_encoder_names(self):
        self.write_zip(self.members())
        report = inventory.inspect_archive(self.archive)
        self.assertTrue(report['expected_names_present'])
        self.assertFalse(report['weights_loaded'])
        self.assertFalse(report['files_extracted'])
        self.assertEqual(len(report['preserved_runs']), 1)

    def test_payloads_are_never_read_or_extracted(self):
        self.write_zip(self.members())
        fail = AssertionError('Attempted to load or extract a payload')
        with patch.object(zipfile.ZipFile, 'read', side_effect=fail), \
             patch.object(zipfile.ZipFile, 'open', side_effect=fail), \
             patch.object(zipfile.ZipFile, 'extractall', side_effect=fail), \
             patch.object(zipfile.ZipFile, 'extract', side_effect=fail):
            self.assertTrue(inventory.inspect_archive(self.archive)['expected_names_present'])

    def test_missing_final_state_is_reported(self):
        self.write_zip([n for n in self.members() if not n.endswith('encoder_final.pt')])
        report = inventory.inspect_archive(self.archive)
        self.assertFalse(report['expected_names_present'])
        self.assertIn(self.prefix + 'encoder_final.pt', report['preserved_runs'][0]['missing_expected_names'])

    def test_missing_root_manifest(self):
        self.write_zip([n for n in self.members() if n != 'export_manifest.json'])
        self.assertEqual(inventory.inspect_archive(self.archive)['missing_root_names'], ['export_manifest.json'])

    def test_saes_are_optional_unless_requested(self):
        self.write_zip(self.members())
        self.assertTrue(inventory.inspect_archive(self.archive)['expected_names_present'])
        self.assertFalse(inventory.inspect_archive(self.archive, require_saes=True)['expected_names_present'])

    def test_expected_sae_names_present(self):
        self.write_zip(self.members(saes=True))
        self.assertTrue(inventory.inspect_archive(self.archive, require_saes=True)['expected_names_present'])

    def test_empty_archive_has_no_completed_inventory(self):
        self.write_zip([])
        self.assertFalse(inventory.inspect_archive(self.archive)['expected_names_present'])

    def test_unsafe_paths_are_rejected(self):
        for name in ['../escape.pt', '/absolute.pt', 'C:/drive.pt', 'folder\\model.pt']:
            with self.subTest(name=name):
                self.write_zip([name.replace('\\', '/')])
                if '\\' in name:
                    # ZipInfo normalizes separators when writing on Windows.
                    # Patch equal-length filename bytes to exercise a foreign ZIP.
                    payload = self.archive.read_bytes().replace(
                        name.replace('\\', '/').encode(), name.encode())
                    self.archive.write_bytes(payload)
                with self.assertRaises(ValueError):
                    inventory.inspect_archive(self.archive)

    def test_symbolic_link_is_rejected(self):
        with zipfile.ZipFile(self.archive, 'w') as archive:
            info = zipfile.ZipInfo('link.pt')
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, 'target.pt')
        with self.assertRaises(ValueError):
            inventory.inspect_archive(self.archive)

    def test_duplicate_names_are_rejected(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            self.write_zip(['protocol.json', 'protocol.json'])
        with self.assertRaises(ValueError):
            inventory.inspect_archive(self.archive)

    def test_invalid_zip_and_missing_path(self):
        self.archive.write_text('not a zip')
        with self.assertRaises(zipfile.BadZipFile):
            inventory.inspect_archive(self.archive)
        with self.assertRaises(FileNotFoundError):
            inventory.inspect_archive(self.folder / 'missing.zip')

    def test_cli_exit_codes_and_read_only_behavior(self):
        self.write_zip(self.members(saes=True))
        before = self.archive.read_bytes()
        result = subprocess.run([sys.executable, str(SCRIPT), str(self.archive), '--require-saes'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)['inventory_only'])
        self.assertEqual(self.archive.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in self.folder.iterdir()), ['run.zip'])
        self.write_zip([])
        result = subprocess.run([sys.executable, str(SCRIPT), str(self.archive)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        result = subprocess.run([sys.executable, str(SCRIPT), str(self.folder/'missing.zip')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)


if __name__ == '__main__':
    unittest.main()
