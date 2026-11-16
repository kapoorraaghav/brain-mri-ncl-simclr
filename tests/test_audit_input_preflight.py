"""Synthetic fingerprint/preflight tests with no patient data or ML runtime."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/check_audit_inputs.py'
spec = importlib.util.spec_from_file_location('audit_preflight', SCRIPT)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        hashes = {}
        self.paths = {}
        for role, (name, key) in preflight.INPUTS.items():
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = ('synthetic fixture for ' + role).encode()
            path.write_bytes(payload)
            hashes[key] = hashlib.sha256(payload).hexdigest()
            self.paths[role] = path
        self.protocol = self.root/preflight.PROTOCOL
        self.protocol.parent.mkdir(parents=True, exist_ok=True)
        self.registration = {'plan': {'version': 'feature-population-audit-v1'}, 'inputs': hashes}
        self.save_protocol()

    def tearDown(self):
        self.temporary.cleanup()

    def save_protocol(self):
        self.protocol.write_text(json.dumps(self.registration), encoding='utf-8')

    def test_matching_registered_inputs(self):
        report = preflight.check_inputs(self.root)
        self.assertTrue(report['checks_passed'])
        self.assertTrue(report['fingerprints_verified'])
        self.assertFalse(report['models_loaded'])
        self.assertEqual([r['fingerprint_status'] for r in report['inputs']], ['match']*3)

    def test_missing_input(self):
        self.paths['historical_saes'].unlink()
        report = preflight.check_inputs(self.root)
        self.assertFalse(report['checks_passed'])
        self.assertEqual(report['inputs'][1]['fingerprint_status'], 'missing')

    def test_hash_mismatch(self):
        self.paths['embeddings'].write_bytes(b'different model family')
        report = preflight.check_inputs(self.root)
        self.assertFalse(report['fingerprints_verified'])
        self.assertEqual(report['inputs'][0]['fingerprint_status'], 'mismatch')

    def test_valid_dry_run_never_hashes_inputs(self):
        with patch.object(preflight, 'sha256', side_effect=AssertionError('Read a private input')):
            report = preflight.check_inputs(self.root, dry_run=True)
        self.assertTrue(report['checks_passed'])
        self.assertFalse(report['fingerprints_verified'])
        self.assertTrue(all(r['fingerprint_status']=='deferred' for r in report['inputs']))

    def test_dry_run_reports_missing_input(self):
        self.paths['covariates'].unlink()
        self.assertFalse(preflight.check_inputs(self.root, dry_run=True)['checks_passed'])

    def test_no_outputs_or_modified_files(self):
        before = {p.relative_to(self.root).as_posix():p.read_bytes()
                  for p in self.root.rglob('*') if p.is_file()}
        preflight.check_inputs(self.root)
        after = {p.relative_to(self.root).as_posix():p.read_bytes()
                 for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse(any(p.name in ('predictions.csv','model.pt') for p in self.root.rglob('*')))

    def test_custom_relative_and_absolute_input_paths(self):
        moved = self.root/'moved.zip'
        self.paths['embeddings'].rename(moved)
        self.assertTrue(preflight.check_inputs(self.root, overrides={'embeddings': 'moved.zip'})['checks_passed'])
        self.assertTrue(preflight.check_inputs(self.root, overrides={'embeddings': moved})['checks_passed'])

    def test_unknown_role(self):
        with self.assertRaises(ValueError):
            preflight.check_inputs(self.root, overrides={'preservation': 'other.zip'})

    def test_preservation_protocol_is_rejected(self):
        self.registration['plan']['version']='preserved-core-rerun-v1'
        self.save_protocol()
        with self.assertRaises(ValueError):
            preflight.check_inputs(self.root)

    def test_invalid_or_missing_hash(self):
        for value in [None, 'not-a-sha256', 123]:
            with self.subTest(value=value):
                self.registration['inputs']['embeddings_sha256']=value
                self.save_protocol()
                with self.assertRaises(ValueError):
                    preflight.check_inputs(self.root)

    def test_missing_or_malformed_protocol(self):
        self.protocol.unlink()
        with self.assertRaises(FileNotFoundError):
            preflight.check_inputs(self.root)
        self.protocol.write_text('{bad json')
        with self.assertRaises(ValueError):
            preflight.check_inputs(self.root)

    def test_cli_modes_and_exit_codes(self):
        def run(*args):
            return subprocess.run([sys.executable,str(SCRIPT),'--root',str(self.root),*args],capture_output=True,text=True)
        result=run('--dry-run')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(json.loads(result.stdout)['fingerprints_verified'])
        result=run()
        self.assertEqual(result.returncode,0,result.stderr)
        self.paths['historical_saes'].write_bytes(b'wrong')
        self.assertEqual(run().returncode,2)
        self.protocol.unlink()
        self.assertEqual(run().returncode,1)


if __name__=='__main__':
    unittest.main()
