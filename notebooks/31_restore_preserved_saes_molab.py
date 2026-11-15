# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24,<1", "torch>=2.7", "torchvision>=0.22", "timm>=1,<2", "numpy>=1.26,<3", "pandas>=2,<4", "scipy>=1.11", "scikit-learn>=1.5,<2", "threadpoolctl>=3"]
# ///
import marimo
__generated_with = "0.24.2"
app = marimo.App(width="full")

@app.cell
def _():
    import marimo as mo
    return (mo,)

@app.cell
def _(mo):
    mo.md("""
    # Restore saved encoder ZIP and finish SAEs

    1. Attach a GPU. Upload **one preserved encoder ZIP** into this runtime's Files panel.
    2. Edit `INPUT_ZIP` below to the **uploaded Molab file path**, not your Windows path.
       The default expects `/marimo/brats_preserved_v1_ncl_seed42_RESUME_SAE.zip`.
    3. Click **Run SAE preparation and export** once. This restores the ZIP, verifies
       weights and embedding IDs, runs 3 SAE seeds (30 epochs each), then CPU probes.
       Existing SAE model weights are reused. **No encoder training, dataset PKL,
       Google Drive login or HF token is needed.**
    4. Wait for **ALL DONE. DOWNLOAD THIS ZIP**. Refresh Files and download the ZIP
       printed at the end. Keep this runtime open until the download is complete.

    Output folder: `/marimo/sae_restore_ncl42`

    Final ZIP: `/marimo/sae_restore_ncl42_WEIGHTS_AND_RESULTS.zip`

    For another encoder ZIP, edit both paths to a new run name. Use one GPU runtime
    per concurrent encoder. CPU probes can be slow even when SAE epochs are fast.
    These are separate reproducibility reruns, not the original paper checkpoints.
    """)
    return

@app.cell
def _():
    # CHANGE THESE TWO LINES for a different uploaded encoder ZIP.
    INPUT_ZIP = "/marimo/brats_preserved_v1_ncl_seed42_RESUME_SAE.zip"
    OUTPUT_DIR = "/marimo/sae_restore_ncl42"
    return INPUT_ZIP, OUTPUT_DIR

@app.cell
def _(INPUT_ZIP, OUTPUT_DIR, mo):
    from pathlib import Path
    _source = Path(INPUT_ZIP)
    mo.md(f"**Input ZIP:** `{INPUT_ZIP}`  \n**Found:** {_source.is_file()}  \n**Output folder:** `{OUTPUT_DIR}`  \n**Final ZIP:** `{OUTPUT_DIR}_WEIGHTS_AND_RESULTS.zip`")
    return (Path,)

@app.cell
def _(mo):
    run_saes = mo.ui.run_button(label="Run SAE preparation and export")
    run_saes
    return (run_saes,)

@app.cell
def _(INPUT_ZIP, OUTPUT_DIR, Path, mo, run_saes):
    mo.stop(not run_saes.value, mo.md("Upload the ZIP, check the input path, then click Run."))
    import subprocess
    import sys
    WORKER_SOURCE = '"""Restore a preserved encoder ZIP and prepare native SAEs without MRI access."""\nfrom pathlib import Path\nimport argparse\nimport hashlib\nimport json\nimport os\nimport shutil\nimport sys\nimport time\nimport types\nimport zipfile\n\n\ndef sha(path):\n    h = hashlib.sha256()\n    with Path(path).open(\'rb\') as stream:\n        for block in iter(lambda: stream.read(8 * 1024 * 1024), b\'\'):\n            h.update(block)\n    return h.hexdigest()\n\n\ndef run(source, output, verify_only=False):\n    source = Path(source).expanduser().resolve()\n    output = Path(output).expanduser().resolve()\n    if not source.is_file():\n        raise FileNotFoundError(f\'Upload the encoder ZIP first. Missing input: {source}\')\n    output.mkdir(parents=True, exist_ok=True)\n    fingerprint = sha(source)\n    receipt_path = output / \'restore_input.json\'\n    if receipt_path.exists():\n        receipt = json.loads(receipt_path.read_text())\n        if receipt[\'zip_sha256\'] != fingerprint:\n            raise ValueError(\'Another input ZIP used this output folder. Choose a new OUTPUT_DIR.\')\n    else:\n        if any(output.iterdir()):\n            raise ValueError(\'Choose an empty output folder for the first restore.\')\n        receipt_path.write_text(json.dumps({\'zip_sha256\': fingerprint,\n                                           \'source_name\': source.name}, indent=2))\n    print(f\'INPUT ZIP: {source}\\nOUTPUT FOLDER: {output}\', flush=True)\n    with zipfile.ZipFile(source) as archive:\n        if archive.testzip() is not None:\n            raise ValueError(\'Input ZIP integrity check failed\')\n        runtimes = [n for n in archive.namelist() if n.endswith(\'portable_runtime.py\')]\n        if len(runtimes) != 1:\n            raise ValueError(\'Upload one preserved encoder run ZIP with portable_runtime.py.\')\n        prefix = runtimes[0][:-len(\'portable_runtime.py\')]\n        for member in archive.infolist():\n            if not member.filename.startswith(prefix):\n                continue\n            relative = member.filename[len(prefix):]\n            if not relative or member.is_dir():\n                continue\n            destination = (output / relative).resolve()\n            if not destination.is_relative_to(output):\n                raise ValueError(\'Unsafe archive member\')\n            if destination.name in [\'STOP\', \'export_manifest.json\']:\n                continue\n            if destination.exists():\n                continue  # Resume files are never overwritten by the input ZIP.\n            destination.parent.mkdir(parents=True, exist_ok=True)\n            temp = destination.with_name(destination.name + \'.restore_tmp\')\n            with archive.open(member) as src, temp.open(\'wb\') as dst:\n                shutil.copyfileobj(src, dst)\n            os.replace(temp, destination)\n    completions = list(output.glob(\'final/*/*/seed_*/completion_verified.json\'))\n    if len(completions) != 1:\n        raise ValueError(\'ZIP must contain exactly one completed encoder run.\')\n    folder = completions[0].parent\n    complete = json.loads(completions[0].read_text())\n    for filename, key in [(\'encoder_final.pt\', \'final_sha256\'), (\'best_loss.pt\', \'best_sha256\')]:\n        if sha(folder / filename) != complete[key]:\n            raise ValueError(f\'Checkpoint hash mismatch: {filename}\')\n    registration = json.loads((output / \'protocol.json\').read_text())\n    method, seed = complete[\'method\'], int(complete[\'seed\'])\n    print(f\'VERIFIED ENCODER: {method}, seed {seed}\', flush=True)\n    if verify_only:\n        print(\'Restore and checkpoint hashes passed. No training performed.\', flush=True)\n        return\n\n    # Execute the archived training recipe, with only a documented numerical\n    # iteration ceiling adjustment. No original encoder is trained or changed.\n    runtime = (output / \'portable_runtime.py\').read_text(encoding=\'utf-8\')\n    old = \'if int(lr.n_iter_.max())>=2000:\'\n    runtime = runtime.replace(old, "if int(lr.n_iter_.max())>=plan[\'probe_max_iter\']:")\n    engine = types.ModuleType(\'restored_sae_runtime\')\n    exec(compile(runtime, \'archived_portable_runtime\', \'exec\'), engine.__dict__)\n    if not engine.torch.cuda.is_available():\n        raise RuntimeError(\'Attach a CUDA GPU to this Molab runtime.\')\n    engine.torch.set_num_threads(2)\n    engine.torch.use_deterministic_algorithms(True)\n    runner = engine.Runner.__new__(engine.Runner)\n    runner.root = output\n    runner.plan = engine.copy.deepcopy(registration[\'plan\'])\n    original_ceiling = runner.plan[\'probe_max_iter\']\n    runner.plan[\'probe_max_iter\'] = max(20000, original_ceiling)\n    runner.registration = registration\n    runner.started = time.monotonic()\n    runner.max_seconds = 72 * 3600\n    runner.backup = None\n    runner.device = engine.torch.device(\'cuda\')\n    runner.core = types.ModuleType(\'restored_core\')\n    exec(compile(engine.CORE_SOURCE, \'archived_core\', \'exec\'), runner.core.__dict__)\n    runner.frame = engine.pd.read_csv(output / \'split_manifest.csv\')\n    manifest = runner.frame.to_csv(index=False, lineterminator=\'\\n\')\n    if hashlib.sha256(manifest.encode()).hexdigest() != registration[\'manifest_sha\']:\n        raise ValueError(\'Manifest does not match registered metadata.\')\n    for split in [\'train\', \'val\']:\n        with engine.np.load(folder / f\'{split}_h_u_z.npz\', allow_pickle=False) as arrays:\n            frame = runner.frame[runner.frame.split == split]\n            if not engine.np.array_equal(arrays[\'sample_ids\'], frame.sample_id.to_numpy(dtype=str)):\n                raise ValueError(\'Embedding sample ordering mismatch\')\n            if not engine.np.isfinite(arrays[\'h\']).all():\n                raise ValueError(\'Nonfinite embeddings\')\n    engine.atomic_json(output / \'restored_sae_execution.json\', {\n        \'source_zip_sha256\': fingerprint, \'method\': method, \'encoder_seed\': seed,\n        \'torch\': str(engine.torch.__version__), \'python\': sys.version,\n        \'gpu\': engine.torch.cuda.get_device_name(0),\n        \'probe_max_iter_original\': original_ceiling,\n        \'probe_max_iter_execution\': runner.plan[\'probe_max_iter\'],\n        \'numerical_override\': \'Convergence ceiling only; C grid, folds, tolerance and objective unchanged\',\n        \'executed_runtime_sha256\': hashlib.sha256(runtime.encode()).hexdigest(),\n        \'encoder_retrained\': False, \'raw_dataset_required\': False, \'test_images_accessed\': False})\n    original_probe = engine.fit_probe\n\n    def visible_probe(*args, **kwargs):\n        print(f\'CPU PROBE START: {args[2]}\', flush=True)\n        started = time.monotonic()\n        result = original_probe(*args, **kwargs)\n        print(f\'CPU PROBE FINISHED: {time.monotonic()-started:.1f}s\', flush=True)\n        return result\n\n    engine.fit_probe = visible_probe\n    print(\'Preparing 3 SAE seeds, 30 epochs each. Saved SAE models are reused.\', flush=True)\n    engine.prepare(runner, [method], [seed])\n    for sae_seed in runner.plan[\'sae_seeds\']:\n        fit = folder / \'evaluation\' / \'native\' / \'baseline\' / f\'sae_{sae_seed}\'\n        for filename in [\'model.pt\', \'validation_full.json\', \'subset_status.json\']:\n            if not (fit / filename).is_file():\n                raise FileNotFoundError(fit / filename)\n        subset = json.loads((fit / \'subset_status.json\').read_text())\n        if subset[\'feasible\'] and not (fit / \'validation_matched64.json\').is_file():\n            raise FileNotFoundError(fit / \'validation_matched64.json\')\n    target = engine.export_preserved(output)\n    engine.atomic_json(output.parent / (output.name + \'_DOWNLOAD_READY.json\'), {\n        \'zip_path\': target, \'zip_sha256\': sha(target),\n        \'status\': \'complete\', \'sae_fits\': len(runner.plan[\'sae_seeds\'])})\n    print(f\'ALL DONE. DOWNLOAD THIS ZIP: {target}\', flush=True)\n    print(\'Download before closing the Molab runtime.\', flush=True)\n\n\nif __name__ == \'__main__\':\n    parser = argparse.ArgumentParser()\n    parser.add_argument(\'--input\', required=True)\n    parser.add_argument(\'--output\', required=True)\n    parser.add_argument(\'--verify-only\', action=\'store_true\')\n    args = parser.parse_args()\n    run(args.input, args.output, args.verify_only)\n'
    _worker = Path(OUTPUT_DIR).parent / (Path(OUTPUT_DIR).name + "_worker.py")
    _worker.parent.mkdir(parents=True, exist_ok=True)
    _worker.write_text(WORKER_SOURCE, encoding="utf-8")
    _log_path = Path(OUTPUT_DIR).parent / (Path(OUTPUT_DIR).name + "_training.log")
    print(f"Log: {_log_path}", flush=True)
    with _log_path.open("a", encoding="utf-8") as _log:
        _process = subprocess.Popen(
            [sys.executable, "-u", str(_worker), "--input", INPUT_ZIP, "--output", OUTPUT_DIR],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=1,
        )
        for _line in _process.stdout:
            print(_line, end="", flush=True)
            _log.write(_line)
            _log.flush()
        _returncode = _process.wait()
    if _returncode:
        raise RuntimeError(f"Worker stopped. Inspect logs above or {_log_path}. Same paths resume saved SAE models.")
    mo.md(f"**Complete. Download:** `{OUTPUT_DIR}_WEIGHTS_AND_RESULTS.zip` from Files.")
    return

if __name__ == "__main__":
    app.run()
