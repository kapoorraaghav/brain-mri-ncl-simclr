"""Read-only preflight for the three registered historical audit inputs.

Reads the published protocol and checks presence and SHA256, without unpickling,
extracting, fitting models or writing outputs. --dry-run checks paths only and
explicitly leaves fingerprint verification deferred. No third-party dependencies.
Exit codes: 0 checks for the requested mode passed; 2 input absent/mismatched;
1 unreadable input or invalid protocol.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    'embeddings': ('data/rank_control_embeddings_v1.zip', 'embeddings_sha256'),
    'historical_saes': ('dai_analysis_reproduction_20261029/data/sae_models.zip',
                        'historical_sae_archive_sha256'),
    'covariates': ('results/reviewer_followup_20261029/confound_audit/validation_covariates.csv',
                   'covariates_sha256'),
}
PROTOCOL = 'results/feature_population_audit_v1/audit_protocol.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def under_root(root, path):
    path = Path(path)
    return (path if path.is_absolute() else root / path).resolve()


def check_inputs(root=ROOT, protocol=PROTOCOL, overrides=None, dry_run=False):
    root = Path(root).resolve()
    overrides = overrides or {}
    unknown = set(overrides) - set(INPUTS)
    if unknown:
        raise ValueError('Unknown input roles: ' + ', '.join(sorted(unknown)))
    protocol_path = under_root(root, protocol)
    registered = json.loads(protocol_path.read_text(encoding='utf-8'))
    if registered.get('plan', {}).get('version') != 'feature-population-audit-v1':
        raise ValueError('Expected the registered historical feature-population audit protocol.')
    hashes = registered.get('inputs', {})
    for _, key in INPUTS.values():
        if not isinstance(hashes.get(key), str) or not re.fullmatch(r'[0-9a-fA-F]{64}', hashes[key]):
            raise ValueError('Missing or invalid protocol SHA256: ' + key)
    rows = []
    for role, (default, key) in INPUTS.items():
        path = under_root(root, overrides.get(role, default))
        present = path.is_file()
        actual = sha256(path) if present and not dry_run else None
        expected = hashes[key].lower()
        status = ('missing' if not present else 'deferred' if dry_run
                  else 'match' if actual == expected else 'mismatch')
        rows.append({'role': role, 'path': str(path), 'present': present,
                     'expected_sha256': expected, 'actual_sha256': actual,
                     'fingerprint_status': status})
    paths_present = all(row['present'] for row in rows)
    verified = not dry_run and all(row['fingerprint_status'] == 'match' for row in rows)
    return {
        'protocol': str(protocol_path), 'model_family': 'registered historical audit',
        'dry_run': bool(dry_run), 'inputs': rows,
        'all_paths_present': paths_present, 'fingerprints_verified': verified,
        'checks_passed': paths_present if dry_run else verified,
        'models_loaded': False, 'predictions_created': False, 'outputs_written': False,
        'notes': ['Preservation reruns are a separate model family, not replacement historical inputs.',
                  'Only the three registered input fingerprints are checked; payload validity, '
                  'all downstream artifacts and scientific reproduction are not established.',
                  'Dry-run mode never hashes patient inputs and does not verify input identities.'],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--protocol', type=Path, default=Path(PROTOCOL))
    parser.add_argument('--embeddings', type=Path)
    parser.add_argument('--sae-models', type=Path)
    parser.add_argument('--covariates', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    overrides = {k: v for k, v in {'embeddings': args.embeddings,
                 'historical_saes': args.sae_models, 'covariates': args.covariates}.items() if v is not None}
    try:
        report = check_inputs(args.root, args.protocol, overrides, args.dry_run)
    except (OSError, ValueError, TypeError, AttributeError) as error:
        print(json.dumps({'error': str(error), 'outputs_written': False}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report['checks_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
