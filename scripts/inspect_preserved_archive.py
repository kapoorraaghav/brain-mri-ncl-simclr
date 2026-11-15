"""Inspect preserved-run ZIP filenames without extracting or loading any payload.

This is an inventory check, not a checkpoint, checksum or inference verification.
Only Python's standard library is required. Exit codes: 0 expected names present;
2 expected files missing; 1 unreadable or unsafe archive metadata.
"""
from pathlib import Path, PurePosixPath, PureWindowsPath
from collections import Counter
import argparse
import json
import stat
import zipfile

ROOT_REQUIRED = ('protocol.json', 'split_manifest.csv', 'export_manifest.json')
RUN_REQUIRED = ('encoder_final.pt', 'best_loss.pt', 'latest.pt', 'history.json',
                'completion_verified.json', 'train_h_u_z.npz', 'val_h_u_z.npz')
RUN_MARKERS = {'encoder_final.pt', 'best_loss.pt', 'latest.pt', 'completion_verified.json'}
SAE_SEEDS = (10, 42, 52)


def inspect_archive(path, require_saes=False):
    """Read the central directory only; never deserialize JSON, PKL or weights."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            raise ValueError('Duplicate member names: ' + ', '.join(duplicates))
        for info in infos:
            # On Windows ZipInfo normalizes backslashes in filename; inspect
            # the original central-directory name before trusting that form.
            name = info.orig_filename
            if (PurePosixPath(name).is_absolute() or PureWindowsPath(name).drive
                    or '\\' in name or '..' in PurePosixPath(name).parts):
                raise ValueError('Unsafe member path: ' + name)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Symbolic-link member: ' + name)
        available = set(names)
        prefixes = sorted({str(PurePosixPath(n).parent) for n in names
                           if PurePosixPath(n).name in RUN_MARKERS})
        runs = []
        for prefix in prefixes:
            base = '' if prefix == '.' else prefix + '/'
            missing = [base + name for name in RUN_REQUIRED if base + name not in available]
            saes = []
            for seed in SAE_SEEDS:
                member = base + f'evaluation/native/baseline/sae_{seed}/model.pt'
                saes.append({'seed': seed, 'path': member, 'present': member in available})
                if require_saes and member not in available:
                    missing.append(member)
            runs.append({'prefix': prefix, 'missing_expected_names': missing,
                         'expected_names_present': not missing, 'native_baseline_saes': saes})
        missing_root = [name for name in ROOT_REQUIRED if name not in available]
        complete = bool(runs) and not missing_root and all(r['expected_names_present'] for r in runs)
        return {
            'archive': str(path), 'inventory_only': True,
            'payloads_read': False, 'weights_loaded': False, 'files_extracted': False,
            'member_count': len(names), 'stored_bytes': sum(i.compress_size for i in infos),
            'declared_uncompressed_bytes': sum(i.file_size for i in infos),
            'missing_root_names': missing_root, 'preserved_runs': runs,
            'checkpoint_members': sorted(n for n in names if n.endswith(('.pt', '.pth', '.safetensors'))),
            'manifest_members': sorted(n for n in names if 'manifest' in PurePosixPath(n).name.lower()
                                       or PurePosixPath(n).name == 'protocol.json'),
            'require_saes': bool(require_saes), 'expected_names_present': complete,
            'notes': ['Member names do not establish valid weights, completed training or reproducible inference.',
                      'No checkpoint fingerprints, ZIP payload CRCs or scientific results were verified.']
                     + ([] if runs else ['No preserved encoder run marker was found.']),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--require-saes', action='store_true',
                        help='Also require native baseline SAE states for seeds 10, 42 and 52.')
    args = parser.parse_args(argv)
    try:
        report = inspect_archive(args.archive, require_saes=args.require_saes)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(json.dumps({'error': str(error), 'inventory_only': True}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report['expected_names_present'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
