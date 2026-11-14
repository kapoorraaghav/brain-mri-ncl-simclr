# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.24,<1", "numpy>=1.26,<3", "pandas>=2,<4",
#   "torch>=2.7", "torchvision>=0.22", "timm>=1.0,<2",
#   "scikit-learn>=1.5,<2", "matplotlib>=3.8,<4", "psutil>=5.9",
#   "fsspec>=2024.6", "gdown>=5.2,<6", "huggingface-hub>=0.34,<3", "python-dotenv>=1,<2",
#   "gdrive-fsspec @ git+https://github.com/marimo-team/gdrive-fsspec",
# ]
# ///
import marimo
__generated_with = "0.24.1"
app = marimo.App(width="full")

@app.cell
def _():
    import marimo as mo
    return (mo,)

@app.cell(hide_code=True)
def _(ENCODER_SEED, METHOD, mo):
    mo.md(f"""
    # {METHOD.upper()} — encoder seed {ENCODER_SEED} — NO CHECKPOINTS

    One encoder followed by six centered SAEs: SAE seeds 10/42/52 at K=24/8.
    Encoder: 50 epochs, LR 0.0005, batch 64. SAE: D=1536, 30 epochs,
    LR 0.0005, batch 512. Training/validation only; no held-out test predictions.

    **No model checkpoints are saved locally or uploaded to Drive. No checkpoint
    authentication is needed. Best-validation encoder weights stay in RAM for
    embeddings and spatial controls, then disappear when the worker exits.**
    JSON histories, metrics, CSV results and embedding caches still save locally.
    Download your CSVs before closing the runtime. Training cannot resume after
    interruption or reset; a fresh run must retrain its encoder and SAEs.

    Dataset: [private Hugging Face repo](https://huggingface.co/datasets/YOUR_USERNAME/YOUR_PRIVATE_DATASET).
    Create a read-only token at https://huggingface.co/settings/tokens and add it
    to each Molab runtime's secrets/environment as **HF_TOKEN**. Do not paste it
    into this notebook. The default source is **huggingface**. The import code
    locates the PKL, downloads it and verifies its size/SHA256 before use.
    Apply settings → **Import / resume dataset** → **Start training**.
    Dataset download can resume; model training cannot. An existing local PKL is
    reused. Optional Drive sources remain available, but are not used by default.
    Hugging Face also has rate limits; this does not promise unlimited downloads.

    Each notebook fixes its encoder seed. Run one notebook per GPU. A fresh output
    path is proposed on opening; choose a new path for another run. Existing
    checkpointed runs and Drive files are not deleted or loaded. There is no
    automatic time-budget stop; provider runtime limits still apply.
    """)
    return

@app.cell(hide_code=True)
def _():
    import textwrap as _textwrap
    METHOD = 'ncl'
    ENCODER_SEED = 10
    CORE_SOURCE = _textwrap.dedent(r'''
    """Seeded BraTS experiments. Existing single-run results are never overwritten."""
    from __future__ import annotations

    import argparse
    import hashlib
    import importlib.metadata
    import json
    import os
    import random
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler


    VERSION = "reviewer-revision-1"


    def json_write(path, obj):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(tmp, path)


    def save_checkpoint(path, payload):
        raise RuntimeError("Model checkpoint saving is disabled in this notebook.")


    def digest(path):
        h = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()


    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def rng_state():
        state = np.random.get_state()
        return {"python": random.getstate(), "torch": torch.get_rng_state(),
                "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


    def restore_rng(state):
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"].cpu())
        n = state["numpy"]
        np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
        if state["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


    def load_config(root, path):
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        if not 1 <= cfg["sae_k"] <= cfg["dictionary_size"]:
            raise ValueError("Require 1 <= SAE K <= dictionary size")
        for key in ("encoder_epochs", "sae_epochs", "batch_size", "sae_batch_size",
                    "top_m", "permutations", "bootstrap_repeats", "selectivity_bootstrap_repeats", "spatial_samples"):
            if cfg[key] < 1:
                raise ValueError(f"{key} must be positive")
        if cfg["batch_size"] < 2:
            raise ValueError("BatchNorm contrastive batches require at least two samples")
        if cfg["num_workers"] != 0:
            raise ValueError("This deterministic Windows runner requires num_workers=0")
        if cfg["train_patch_slice_limit"] < 1 or cfg["auxk"] < 1 or cfg["dead_after_samples"] < 1:
            raise ValueError("Patch limit, AuxK and dead-feature threshold must be positive")
        if set(cfg["modes"]) - {"supervised", "simclr", "ncl"}:
            raise ValueError("Unknown encoder mode")
        if set(cfg["token_spaces"]) - {"cls", "patch"}:
            raise ValueError("Unknown token space")
        if set(cfg["sae_variants"]) - {"baseline", "auxk"}:
            raise ValueError("Unknown SAE variant")
        cfg["root"] = str(Path(root).resolve())
        cfg["version"] = VERSION
        cfg["source_sha256"] = digest(__file__)
        return cfg


    def metadata(cfg):
        root = Path(cfg["root"])
        path = root / cfg["metadata"]
        df = pd.read_csv(path)
        required = {"patient_id", "slice_idx", "split", "file_path", "dominant_label", "has_tumor"}
        if required - set(df.columns):
            raise ValueError(f"Missing metadata columns: {required - set(df.columns)}")
        if df.empty or set(df.split) != {"train", "val", "test"}:
            raise ValueError("Require nonempty train, val and test partitions")
        if (df.groupby("patient_id").split.nunique() != 1).any():
            raise ValueError("Patient leakage between splits")
        if df.duplicated(["patient_id", "slice_idx"]).any():
            raise ValueError("Duplicate patient/slice records")
        if not df.dominant_label.isin([0, 1, 2, 4]).all():
            raise ValueError("Unexpected segmentation label")
        if not np.array_equal(df.has_tumor.to_numpy(), (df.dominant_label != 0).astype(int)):
            raise ValueError("Tumor labels disagree")
        # Resolve both portable paths and the existing machine-specific metadata.
        def resolve(row):
            p = Path(row.file_path)
            candidates = [p if p.is_absolute() else root / p,
                          path.parent / row.split / p.name]
            found = next((x for x in candidates if x.is_file()), None)
            if found is None:
                raise FileNotFoundError(row.file_path)
            return str(found.absolute())
        df["file_path"] = df.apply(resolve, axis=1)
        df["sample_id"] = df.patient_id.astype(str) + ":" + df.slice_idx.astype(str)
        return df


    def establish_manifest(cfg, df):
        out = Path(cfg["root"]) / cfg["output"]
        out.mkdir(parents=True, exist_ok=True)
        manifest = df[["patient_id", "slice_idx", "split", "dominant_label", "has_tumor", "sample_id"]]
        fingerprint = hashlib.sha256(manifest.to_csv(index=False).encode()).hexdigest()
        cfg["metadata_sha256"] = digest(Path(cfg["root"]) / cfg["metadata"])
        cfg["manifest_sha256"] = fingerprint
        old = out / "config.json"
        if old.exists() and json.loads(old.read_text()) != cfg:
            raise ValueError("Configuration/source changed: choose a new output directory")
        json_write(old, cfg)
        manifest.to_csv(out / "split_manifest.csv", index=False)
        json_write(out / "environment.json", {"python": sys.version, "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "cuda": torch.version.cuda, "device": str(device()), "packages": {
                name: importlib.metadata.version(name) for name in
                ["torch", "torchvision", "timm", "numpy", "pandas", "scikit-learn"]}})
        json_write(out / "class_frequencies.json", {
            split: {"patients": int(g.patient_id.nunique()), "slices": len(g),
                    "dominant_labels": {str(k): int(v) for k, v in g.dominant_label.value_counts().items()}}
            for split, g in df.groupby("split")})
        return out


    def device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


    class Slices(Dataset):
        def __init__(self, df, cfg, mode=None):
            self.df = df.reset_index(drop=True)
            self.mode = mode
            self.transform = None
            if mode:
                import torchvision.transforms.v2 as T
                ops = ([T.RandomResizedCrop(cfg["img_size"], scale=(0.7, 1.0)),
                        T.RandomHorizontalFlip(0.5), T.RandomRotation(15), T.GaussianNoise(sigma=0.03)]
                       if mode != "supervised" else [T.RandomHorizontalFlip(0.5), T.RandomRotation(10)])
                self.transform = T.Compose(ops)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, index):
            row = self.df.iloc[index]
            # Input dictionaries were produced by this project's preprocessing script.
            item = load_item(row.file_path)
            image = torch.from_numpy(item["image"]).float().unsqueeze(0)
            if not torch.isfinite(image).all():
                raise ValueError(f"Nonfinite image: {row.sample_id}")
            if self.mode and self.mode != "supervised":
                return self.transform(image), self.transform(image)
            return (self.transform(image) if self.transform else image), int(row.has_tumor)


    class Encoder(nn.Module):
        def __init__(self, cfg, mode):
            super().__init__()
            import timm
            self.mode = mode
            self.backbone = timm.create_model("vit_tiny_patch16_224", pretrained=False,
                img_size=cfg["img_size"], in_chans=1, num_classes=0)
            self.head = (nn.Linear(192, 2) if mode == "supervised" else
                         nn.Sequential(nn.Linear(192, 192), nn.BatchNorm1d(192), nn.ReLU(), nn.Linear(192, 128)))

        def forward(self, x):
            z = self.head(self.backbone(x))
            return F.relu(z) if self.mode == "ncl" else z


    def contrastive_loss(z1, z2, temperature):
        n = len(z1)
        z = F.normalize(torch.cat([z1, z2]).float(), dim=1, eps=1e-8)
        logits = z @ z.T / temperature
        logits.fill_diagonal_(float("-inf"))
        targets = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
        return F.cross_entropy(logits, targets)


    def validation_objective(model, frame, cfg):
        # Fixed augmentation stream at every epoch; test data never participates.
        previous = rng_state()
        seed_all(123456)
        loader = DataLoader(Slices(frame, cfg, None if model.mode == "supervised" else model.mode),
                            batch_size=cfg["batch_size"], shuffle=False, drop_last=model.mode != "supervised")
        total, count = 0., 0
        model.eval()
        try:
            with torch.no_grad():
                for a, b in loader:
                    a, b = a.to(device()), b.to(device())
                    loss = (F.cross_entropy(model(a), b) if model.mode == "supervised" else
                            contrastive_loss(model(a), model(b), cfg["temperature"]))
                    total += loss.item() * len(a)
                    count += len(a)
        finally:
            restore_rng(previous)
        if not count:
            raise ValueError("Validation loader is empty")
        return total / count


    def restore_scaler(scaler, saved_state):
        """Disabled AMP saves {}; it cannot initialize an enabled GradScaler."""
        if scaler.is_enabled() and saved_state:
            scaler.load_state_dict(saved_state)
        elif scaler.is_enabled():
            print("Checkpoint has no AMP scaler state; using a fresh CUDA scaler while resuming model and optimizer.", flush=True)


    def train_encoder(cfg, df, out, mode, seed):
        run = out / mode / f"encoder_{seed}"
        seed_all(seed)
        model = Encoder(cfg, mode).to(device())
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["encoder_lr"], weight_decay=cfg["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=device().type == "cuda")
        history, start, best_loss = [], 0, float("inf")
        pass
        loader = DataLoader(Slices(df[df.split == "train"], cfg, mode), batch_size=cfg["batch_size"],
                            shuffle=True, drop_last=True, num_workers=cfg["num_workers"])
        if not len(loader):
            raise ValueError("Training loader is empty")
        for epoch in range(start, cfg["encoder_epochs"]):
            check_budget()
            model.train()
            losses, zeros = [], []
            for a, b in loader:
                a, b = a.to(device()), b.to(device())
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device().type, enabled=device().type == "cuda"):
                    if mode == "supervised":
                        loss = F.cross_entropy(model(a), b)
                    else:
                        z1, z2 = model(a), model(b)
                        loss = contrastive_loss(z1, z2, cfg["temperature"])
                        zeros.append(float((z1.float().norm(dim=1) == 0).float().mean()))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite encoder loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
            val_loss = validation_objective(model, df[df.split == "val"], cfg)
            improved = val_loss < best_loss
            best_loss = min(best_loss, val_loss)
            history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val_loss,
                            "zero_projection_fraction": float(np.mean(zeros)) if zeros else None})
            pass
            if improved:
                retain_best_model(mode, seed, model)
            json_write(run / "history.json", history)
            print(f"{mode} encoder={seed} epoch={epoch+1} train={np.mean(losses):.4f} val={val_loss:.4f}", flush=True)
        return run


    def extraction(cfg, model, frame, run, split, space):
        frame = frame.reset_index(drop=True)
        if split == "train" and space == "patch":
            frame = frame.sample(n=min(len(frame), cfg["train_patch_slice_limit"]), random_state=123456).sort_index()
        identity = {"checkpoint": model_digest(run.parent.name, int(run.name.removeprefix("encoder_"))), "manifest": cfg["manifest_sha256"],
                    "space": space, "split": split, "version": VERSION,
                    "samples": hashlib.sha256("\n".join(frame.sample_id).encode()).hexdigest()}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        cache = run / "embeddings" / f"{split}_{space}_{key}.npz"
        if cache.exists():
            with np.load(cache) as data:
                return data["x"], frame
        model.eval()
        collected = []
        loader = DataLoader(Slices(frame, cfg), batch_size=cfg["batch_size"], shuffle=False)
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device())
                value = (model.backbone(images) if space == "cls" else
                         model.backbone.forward_features(images)[:, 1:, :])
                collected.append(value.float().cpu().numpy())
        x = np.concatenate(collected)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite representations")
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        with open(tmp, "wb") as stream:
            np.savez(stream, x=x, sample_ids=frame.sample_id.to_numpy(dtype=str))
        os.replace(tmp, cache)
        json_write(cache.with_suffix(".json"), identity)
        return x, frame


    class TopKSAE(nn.Module):
        def __init__(self, width, dictionary, k):
            super().__init__()
            self.k = k
            self.pre_bias = nn.Parameter(torch.zeros(width))
            self.encoder = nn.Linear(width, dictionary)
            self.decoder = nn.Linear(dictionary, width, bias=False)
            self.register_buffer("inactive_samples", torch.zeros(dictionary, dtype=torch.long))

        @staticmethod
        def sparsify(values, k):
            v, i = values.topk(k, dim=-1)
            return torch.zeros_like(values).scatter_(-1, i, v)

        def forward(self, x):
            raw = F.relu(self.encoder(x - self.pre_bias))
            acts = self.sparsify(raw, self.k)
            return self.decoder(acts) + self.pre_bias, acts, raw

        def auxiliary_loss(self, x, recon, acts, raw, cfg):
            # AuxK-style residual reconstruction by currently dead units; not a
            # reproduction of Gao et al.'s large-language-model training recipe.
            with torch.no_grad():
                self.inactive_samples.add_(len(x))
                self.inactive_samples[(acts > 0).any(dim=0)] = 0
            dead = self.inactive_samples >= cfg["dead_after_samples"]
            k = min(cfg["auxk"], int(dead.sum()))
            if not k:
                return x.new_zeros(())
            candidate = raw.masked_fill(~dead.unsqueeze(0), 0.)
            residual_hat = self.decoder(self.sparsify(candidate, k))
            return F.mse_loss(residual_hat, (x - recon).detach())


    def train_sae(cfg, x, folder, seed, variant):
        seed_all(seed)
        x = x.reshape(-1, x.shape[-1])
        sae = TopKSAE(x.shape[-1], cfg["dictionary_size"], cfg["sae_k"]).to(device())
        optimizer = torch.optim.Adam(sae.parameters(), lr=cfg["sae_lr"])
        start, history = 0, []
        latest = folder / "sae.pt"
        if latest.exists():
            state = torch.load(latest, map_location=device(), weights_only=True)
            sae.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start, history = state["epoch"] + 1, state["history"]
            restore_rng(state["rng"])
        loader = DataLoader(TensorDataset(torch.from_numpy(x).float()), batch_size=cfg["sae_batch_size"], shuffle=True)
        for epoch in range(start, cfg["sae_epochs"]):
            total, auxiliary, count = 0., 0., 0
            sae.train()
            for (batch,) in loader:
                batch = batch.to(device())
                optimizer.zero_grad(set_to_none=True)
                recon, acts, raw = sae(batch)
                mse = F.mse_loss(recon, batch)
                aux = sae.auxiliary_loss(batch, recon, acts, raw, cfg) if variant == "auxk" else batch.new_zeros(())
                loss = mse + cfg["aux_weight"] * aux
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite SAE loss")
                loss.backward()
                nn.utils.clip_grad_norm_(sae.parameters(), 1.)
                optimizer.step()
                total += mse.item() * len(batch)
                auxiliary += aux.item() * len(batch)
                count += len(batch)
            history.append({"epoch": epoch + 1, "mse": total/count, "aux_loss": auxiliary/count})
            save_checkpoint(latest, {"model": sae.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "history": history, "rng": rng_state(), "sae_seed": seed,
                "variant": variant, "config": cfg})
            json_write(folder / "history.json", history)
            print(f"SAE {folder.name} seed={seed} epoch={epoch+1} mse={total/count:.6g}", flush=True)
        sae.eval()
        return sae


    def representation_diagnostics(x):
        x = np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])
        centered = x - x.mean(axis=0)
        eigenvalues = np.maximum(np.linalg.eigvalsh(centered.T @ centered / max(len(x)-1, 1)), 0.)
        total = eigenvalues.sum()
        p = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
        positive = p[p > 0]
        return {"total_variance": float(total), "effective_rank": float(np.exp(-(positive*np.log(positive)).sum())) if total else 0.,
                "participation_ratio": float(total**2 / np.sum(eigenvalues**2)) if total else 0.,
                "covariance_eigenvalues": eigenvalues.tolist()}


    def sae_outputs(sae, x, batch_size=512):
        shape = x.shape
        xflat = x.reshape(-1, shape[-1])
        recon, acts = [], []
        with torch.no_grad():
            for start in range(0, len(xflat), batch_size):
                r, a, _ = sae(torch.from_numpy(xflat[start:start+batch_size]).float().to(device()))
                recon.append(r.cpu().numpy())
                acts.append(a.cpu().numpy())
        return np.concatenate(recon).reshape(shape), np.concatenate(acts).reshape(*shape[:-1], sae.encoder.out_features)


    def reconstruction_metrics(x, recon, train_mean):
        residual = float(np.sum((x.astype(np.float64) - recon)**2))
        reference = float(np.sum((x.astype(np.float64) - train_mean)**2))
        ratio = residual/reference if reference > 0 else None
        return {"mse": residual/x.size, "normalized_mse_train_mean": ratio,
                "variance_explained_train_mean": 1-ratio if ratio is not None else None}


    def evaluate_sae_stream(sae, x, train_mean):
        """Pool patch codes per slice without allocating N*64*D activations."""
        pooled, residual, reference = [], 0., 0.
        batch_size = 16 if x.ndim == 3 else 512
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                batch = x[start:start+batch_size]
                shape = batch.shape
                flat = torch.from_numpy(batch.reshape(-1, shape[-1])).float().to(device())
                recon, acts, _ = sae(flat)
                recon = recon.cpu().numpy().reshape(shape)
                acts = acts.reshape(*shape[:-1], sae.encoder.out_features)
                pooled.append((acts.mean(dim=1) if batch.ndim == 3 else acts).cpu().numpy())
                residual += float(np.sum((batch.astype(np.float64)-recon)**2))
                reference += float(np.sum((batch.astype(np.float64)-train_mean)**2))
        ratio = residual/reference if reference > 0 else None
        return np.concatenate(pooled), {"mse": residual/x.size, "normalized_mse_train_mean": ratio,
            "variance_explained_train_mean": 1-ratio if ratio is not None else None}


    def selectivity(acts, labels, m, permutations, seed):
        # Eligibility requires M genuinely positive examples for every feature.
        support = (acts > 0).sum(axis=0)
        eligible = np.where(support >= m)[0]
        alive = np.where(support > 0)[0]
        top = np.argsort(acts[:, eligible], axis=0)[-m:].T if len(eligible) else np.empty((0, m), int)
        def scores(y):
            purity, entropy, majority = [], [], []
            for indices in top:
                values, counts = np.unique(y[indices], return_counts=True)
                probabilities = counts/m
                purity.append(float(probabilities.max()))
                entropy.append(float(-(probabilities*np.log(probabilities)).sum()))
                majority.append(int(values[counts.argmax()]))
            return purity, entropy, majority
        purities, entropies, majorities = scores(labels)
        rng = np.random.default_rng(seed)
        null = []
        for _ in range(permutations):
            p, e, _ = scores(rng.permutation(labels))
            if p:
                null.append([np.mean(p), np.mean(e)])
        null = np.asarray(null)
        metrics = {"alive_features": len(alive), "eligible_features": len(eligible),
            "under_supported_features": int(((support > 0) & (support < m)).sum()),
            "mean_purity": float(np.mean(purities)) if purities else None,
            "mean_entropy_nats": float(np.mean(entropies)) if entropies else None,
            "mean_positive_features_per_slice": float((acts > 0).sum(axis=1).mean()),
            "alive_fraction_used_per_slice": float((acts > 0).sum(axis=1).mean()/len(alive)) if len(alive) else None,
            "permutation_mean_purity": float(null[:, 0].mean()) if len(null) else None,
            "permutation_purity_95_interval": np.quantile(null[:, 0], [.025, .975]).tolist() if len(null) else None,
            "permutation_mean_entropy_nats": float(null[:, 1].mean()) if len(null) else None,
            "permutation_scope": "slice-label shuffle; descriptive reference, not a patient-independent significance test"}
        table = pd.DataFrame({"feature": eligible, "positive_support": support[eligible],
                              "purity": purities, "entropy_nats": entropies, "majority_label": majorities})
        return metrics, table, top


    def patient_bootstrap(correct, patients, repeats, seed):
        # Keep every slice of each sampled patient together, including duplicates.
        frame = pd.DataFrame({"correct": correct, "patient": patients})
        g = frame.groupby("patient").correct.agg(["sum", "count"])
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(g), (repeats, len(g)))
        values = g["sum"].to_numpy()[draws].sum(axis=1)/g["count"].to_numpy()[draws].sum(axis=1)
        return {"slice_accuracy": float(np.mean(correct)),
                "patient_macro_accuracy": float((g["sum"]/g["count"]).mean()),
                "patient_bootstrap_slice_accuracy_95_ci": np.quantile(values, [.025, .975]).tolist(),
                "patients": len(g)}


    def selectivity_bootstrap(acts, labels, patients, m, repeats, seed):
        """Resample whole patients and recompute top-M and feature eligibility."""
        groups = [np.where(patients == p)[0] for p in np.unique(patients)]
        rng = np.random.default_rng(seed)
        classes = np.unique(labels)
        estimates = []
        for _ in range(repeats):
            indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            resampled = acts[indices]
            eligible = np.where((resampled > 0).sum(axis=0) >= m)[0]
            if not len(eligible):
                continue
            values = resampled[:, eligible]
            top = np.argpartition(values, len(values)-m, axis=0)[-m:]
            selected_labels = labels[indices][top]
            probabilities = np.stack([(selected_labels == c).mean(axis=0) for c in classes])
            entropy = -(probabilities*np.log(np.maximum(probabilities, 1e-12))).sum(axis=0)
            estimates.append([probabilities.max(axis=0).mean(), entropy.mean()])
        array = np.asarray(estimates)
        return {"patient_bootstrap_purity_95_ci": np.quantile(array[:, 0], [.025, .975]).tolist() if len(array) else None,
                "patient_bootstrap_entropy_nats_95_ci": np.quantile(array[:, 1], [.025, .975]).tolist() if len(array) else None,
                "selectivity_bootstrap_valid_repeats": len(array),
                "selectivity_bootstrap_scope": "conditional on this fitted dictionary; patient resampling, top-M and support recomputed"}


    def spatial_audit(cfg, model, sae, frame, eligible, space):
        # All models use identical sample IDs; inactive features contribute zero Dice.
        frame = frame[frame.has_tumor == 1].sample(n=min(cfg["spatial_samples"], int((frame.has_tumor == 1).sum())), random_state=123456)
        scores, coverage = [], []
        grid = cfg["img_size"]//16
        with torch.no_grad():
            for row in frame.itertuples():
                item = load_item(row.file_path)
                image = torch.from_numpy(item["image"]).float()[None, None].to(device())
                patches = model.backbone.forward_features(image)[:, 1:, :].squeeze(0)
                if len(patches) != grid*grid:
                    raise ValueError("Unexpected token grid")
                _, acts, _ = sae(patches)
                acts = acts.cpu().numpy()[:, eligible]
                threshold = np.percentile(acts, cfg["spatial_percentile"], axis=0)
                predicted = (acts > threshold) & (acts > 0)
                mask = item["seg_mask"] > 0
                truth = np.array([mask[i*16:(i+1)*16,j*16:(j+1)*16].mean() > cfg["tumor_patch_fraction"]
                                  for i in range(grid) for j in range(grid)])
                intersection = (predicted & truth[:, None]).sum(axis=0)
                denom = predicted.sum(axis=0) + truth.sum()
                dice = np.divide(2*intersection, denom, out=np.zeros(len(eligible), float), where=denom > 0)
                scores.append(dice)
                coverage.append((acts > 0).any(axis=0))
        scores = np.asarray(scores).reshape(len(frame), len(eligible))
        coverage = np.asarray(coverage).reshape(len(frame), len(eligible))
        table = pd.DataFrame({"feature": eligible,
            "mean_dice_all_sampled_slices": scores.mean(axis=0) if len(frame) else np.zeros(len(eligible)),
            "activation_coverage": coverage.mean(axis=0) if len(frame) else np.zeros(len(eligible))})
        return table, {"sample_ids": frame.sample_id.tolist(), "sample_count": len(frame),
            "grid_size": grid, "inactive_policy": "zero Dice; never omitted",
            "sae_training_space": space, "patch_application": "direct" if space == "patch" else "CLS-to-patch transfer control"}


    def audit(cfg, df, out, mode, seed, split):
        run = out / mode / f"encoder_{seed}"
        if split == "test":
            for space in cfg["token_spaces"]:
                for variant in cfg["sae_variants"]:
                    for sae_seed in cfg["sae_seeds"]:
                        if not (run / f"sae_{space}_{variant}_{sae_seed}" / "val/selection.json").exists():
                            raise ValueError("Run validation audit before test; feature selection must be frozen")
        latest = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
        if latest["epoch"] + 1 != cfg["encoder_epochs"]:
            raise ValueError("Complete encoder training before auditing")
        state = torch.load(run / "best.pt", map_location=device(), weights_only=True)
        model = Encoder(cfg, mode).to(device())
        model.load_state_dict(state["model"])
        model.eval()
        train_cls, train_frame = extraction(cfg, model, df[df.split == "train"], run, "train", "cls")
        eval_cls, eval_frame = extraction(cfg, model, df[df.split == split], run, split, "cls")
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1., random_state=seed))
        probe.fit(train_cls, train_frame.has_tumor)
        predictions = probe.predict(eval_cls)
        correct = predictions == eval_frame.has_tumor.to_numpy()
        metrics = patient_bootstrap(correct, eval_frame.patient_id.to_numpy(), cfg["bootstrap_repeats"], seed)
        metrics["representation"] = representation_diagnostics(eval_cls)
        folder = run / split
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"sample_id": eval_frame.sample_id, "patient_id": eval_frame.patient_id,
            "label": eval_frame.has_tumor, "prediction": predictions, "correct": correct}).to_csv(folder / "probe_predictions.csv", index=False)
        json_write(folder / "encoder_metrics.json", metrics)
        if mode == "supervised":
            head_predictions = []
            with torch.no_grad():
                for images, _ in DataLoader(Slices(eval_frame, cfg), batch_size=cfg["batch_size"]):
                    head_predictions.extend(model(images.to(device())).argmax(dim=1).cpu().tolist())
            json_write(folder / "supervised_head_metrics.json", patient_bootstrap(
                np.array(head_predictions) == eval_frame.has_tumor.to_numpy(), eval_frame.patient_id.to_numpy(), cfg["bootstrap_repeats"], seed))
        for space in cfg["token_spaces"]:
            train_x, _ = extraction(cfg, model, df[df.split == "train"], run, "train", space)
            eval_x, eval_frame = extraction(cfg, model, df[df.split == split], run, split, space)
            train_mean = train_x.reshape(-1, 192).mean(axis=0)
            for variant in cfg["sae_variants"]:
                for sae_seed in cfg["sae_seeds"]:
                    sae_folder = run / f"sae_{space}_{variant}_{sae_seed}"
                    if split == "test" and not (sae_folder / "val" / "selection.json").exists():
                        raise ValueError("Run validation audit before test; feature selection must be frozen")
                    if split == "val":
                        sae = train_sae(cfg, train_x, sae_folder, sae_seed, variant)
                    else:
                        saved = torch.load(sae_folder / "sae.pt", map_location=device(), weights_only=True)
                        sae = TopKSAE(192, cfg["dictionary_size"], cfg["sae_k"]).to(device())
                        sae.load_state_dict(saved["model"])
                        sae.eval()
                    slice_acts, reconstruction = evaluate_sae_stream(sae, eval_x, train_mean)
                    result, features, top = selectivity(slice_acts, eval_frame.dominant_label.to_numpy(),
                        cfg["top_m"], cfg["permutations"], sae_seed)
                    result.update(reconstruction)
                    result.update(selectivity_bootstrap(slice_acts, eval_frame.dominant_label.to_numpy(),
                        eval_frame.patient_id.to_numpy(), cfg["top_m"], cfg["selectivity_bootstrap_repeats"], sae_seed))
                    result.update({"split": split, "mode": mode, "encoder_seed": seed,
                        "sae_seed": sae_seed, "space": space, "variant": variant,
                        "dictionary_size": cfg["dictionary_size"], "k": cfg["sae_k"]})
                    destination = sae_folder / split
                    destination.mkdir(parents=True, exist_ok=True)
                    if split == "val":
                        ordered = features.sort_values(["purity", "positive_support", "feature"], ascending=[False, False, True])
                        selection = {"feature": int(ordered.iloc[0].feature) if len(ordered) else None,
                            "rule": "validation purity, then positive support, then feature index",
                            "spatial_features": np.where((slice_acts > 0).any(axis=0))[0].tolist()}
                        json_write(destination / "selection.json", selection)
                    else:
                        selection = json.loads((sae_folder / "val" / "selection.json").read_text())
                    dice, spatial = spatial_audit(cfg, model, sae, eval_frame,
                                                 np.asarray(selection["spatial_features"], dtype=int), space)
                    selected = dice[dice.feature == selection["feature"]]
                    result["validation_selected_feature"] = selection["feature"]
                    result["selected_feature_dice"] = float(selected.iloc[0].mean_dice_all_sampled_slices) if len(selected) else None
                    result["mean_feature_dice"] = float(dice.mean_dice_all_sampled_slices.mean()) if len(dice) else None
                    features.to_csv(destination / "feature_metrics.csv", index=False)
                    dice.to_csv(destination / "spatial_feature_metrics.csv", index=False)
                    examples = []
                    for feature, indices in zip(features.feature, top):
                        for idx in indices:
                            r = eval_frame.iloc[idx]
                            examples.append({"feature": int(feature), "sample_id": r.sample_id,
                                             "patient_id": r.patient_id, "dominant_label": int(r.dominant_label)})
                    pd.DataFrame(examples, columns=["feature", "sample_id", "patient_id", "dominant_label"]).to_csv(destination / "top_examples.csv", index=False)
                    json_write(destination / "spatial_protocol.json", spatial)
                    json_write(destination / "metrics.json", result)
                    print(f"Audited {mode}/{seed}/{space}/{variant}/{sae_seed}/{split}", flush=True)


    def summary(out, split):
        cfg = json.loads((out / "config.json").read_text())
        rows = [json.loads(p.read_text()) for p in out.glob(f"*/encoder_*/sae_*/{split}/metrics.json")]
        rows = [r for r in rows if r["mode"] in cfg["modes"]
                and r["encoder_seed"] in cfg["encoder_seeds"] and r["sae_seed"] in cfg["sae_seeds"]
                and r["space"] in cfg["token_spaces"] and r["variant"] in cfg["sae_variants"]]
        if not rows:
            raise ValueError("No completed SAE audits")
        frame = pd.DataFrame(rows)
        frame.to_csv(out / f"{split}_all_sae_runs.csv", index=False)
        columns = ["mean_purity", "mean_entropy_nats", "alive_features", "normalized_mse_train_mean",
                   "variance_explained_train_mean", "mean_feature_dice", "selected_feature_dice"]
        # First average SAE replicates within each encoder; 27 dictionaries are not
        # 27 independent encoder replications.
        per_encoder = frame.groupby(["mode", "space", "variant", "encoder_seed"])[columns].mean()
        per_encoder.to_csv(out / f"{split}_per_encoder.csv")
        per_encoder.groupby(["mode", "space", "variant"]).agg(["mean", "std", "count"]).to_csv(out / f"{split}_summary.csv")
        probe_rows = []
        for path in out.glob(f"*/encoder_*/{split}/encoder_metrics.json"):
            if path.parents[2].name not in cfg["modes"] or int(path.parents[1].name.split("_")[-1]) not in cfg["encoder_seeds"]:
                continue
            m = json.loads(path.read_text())
            probe_rows.append({"mode": path.parents[2].name, "encoder_seed": int(path.parents[1].name.split("_")[-1]),
                               "accuracy": m["slice_accuracy"], "effective_rank": m["representation"]["effective_rank"]})
        probes = pd.DataFrame(probe_rows)
        probes.to_csv(out / f"{split}_probe_runs.csv", index=False)
        probes.groupby("mode")[["accuracy", "effective_rank"]].agg(["mean", "std", "count"]).to_csv(out / f"{split}_probe_summary.csv")
        # Paired patient bootstrap of NCL - SimCLR for each matched encoder seed.
        for seed in sorted(probes.encoder_seed.unique()):
            a = out / "ncl" / f"encoder_{seed}" / split / "probe_predictions.csv"
            b = out / "simclr" / f"encoder_{seed}" / split / "probe_predictions.csv"
            if not a.exists() or not b.exists():
                continue
            ncl, sim = pd.read_csv(a), pd.read_csv(b)
            joined = ncl.merge(sim, on=["sample_id", "patient_id", "label"], suffixes=("_ncl", "_sim"), validate="one_to_one")
            if len(joined) != len(ncl) or len(joined) != len(sim):
                raise ValueError("Paired probe samples do not match")
            difference = joined.correct_ncl.astype(int) - joined.correct_sim.astype(int)
            cfg = json.loads((out / "config.json").read_text())
            result = patient_bootstrap(difference, joined.patient_id, cfg["bootstrap_repeats"], int(seed))
            json_write(out / f"{split}_paired_probe_seed_{seed}.json", {
                "ncl_minus_simclr_accuracy": result["slice_accuracy"],
                "patient_bootstrap_95_ci": result["patient_bootstrap_slice_accuracy_95_ci"],
                "patients": result["patients"], "conditional_on_encoder_seed": int(seed)})


    def main():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
        parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment_config.json"))
        parser.add_argument("--stage", choices=["check", "train", "validate", "test", "summarize"], default="check")
        parser.add_argument("--mode", choices=["supervised", "simclr", "ncl"])
        parser.add_argument("--encoder-seed", type=int)
        parser.add_argument("--split", choices=["val", "test"], default="val")
        args = parser.parse_args()
        cfg = load_config(args.root, args.config)
        df = metadata(cfg)
        out = establish_manifest(cfg, df)
        modes = [args.mode] if args.mode else cfg["modes"]
        seeds = [args.encoder_seed] if args.encoder_seed is not None else cfg["encoder_seeds"]
        if any(m not in cfg["modes"] for m in modes) or any(s not in cfg["encoder_seeds"] for s in seeds):
            raise ValueError("Requested mode or seed is absent from the registered configuration")
        if args.stage == "check":
            print(f"Validated {len(df)} slices; {len(seeds)*len(modes)} encoder runs, "
                  f"{len(seeds)*len(modes)*len(cfg['sae_seeds'])*len(cfg['sae_variants'])*len(cfg['token_spaces'])} SAE fits")
        elif args.stage == "summarize":
            summary(out, args.split)
        else:
            for mode in modes:
                for seed in seeds:
                    if args.stage == "train":
                        train_encoder(cfg, df, out, mode, seed)
                    else:
                        audit(cfg, df, out, mode, seed, "val" if args.stage == "validate" else "test")


    if __name__ == "__main__":
        main()
    ''').lstrip('\n')
    SAE_SOURCE = _textwrap.dedent(r'''
    def write_json(path, value):
        exp.json_write(path, value)

    def evaluate_cls_stream(model,x,train_mean):
        # Follow the model's actual device; diagnostic CPU and pilot CUDA are both supported.
        device=next(model.parameters()).device
        codes=[]; residual=reference=0.
        with torch.no_grad():
            for b in range(0,len(x),512):
                batch=x[b:b+512]
                recon,acts,_=model(torch.from_numpy(batch).float().to(device))
                codes.append(acts.cpu().numpy())
                residual+=float(np.sum((batch.astype(np.float64)-recon.cpu().numpy().astype(np.float64))**2))
                reference+=float(np.sum((batch.astype(np.float64)-train_mean)**2))
        assert reference>0,'Zero validation variance against training mean'
        return np.concatenate(codes),{'mse':residual/x.size,'normalized_mse_train_mean':residual/reference,
                                    'variance_explained_train_mean':1-residual/reference}

    def cached_cls(mode, seed, split):
        assert split in ('train', 'val')
        run = SOURCE/mode/f'encoder_{seed}'
        expected_ids = MANIFEST.loc[MANIFEST.split == split, 'sample_id'].astype(str).to_numpy()
        expected_samples = hashlib.sha256('\n'.join(expected_ids).encode()).hexdigest()
        checkpoint_hash = exp.model_digest(mode, seed)
        candidates = []
        for meta_path in (run/'embeddings').glob(f'{split}_cls_*.json'):
            identity = json.loads(meta_path.read_text())
            if (identity.get('checkpoint') == checkpoint_hash and identity.get('manifest') == CFG['manifest_sha256']
                and identity.get('samples') == expected_samples and identity.get('space') == 'cls'
                and identity.get('split') == split and identity.get('version') == CFG['version']):
                candidates.append(meta_path.with_suffix('.npz'))
        assert len(candidates) == 1, f'Expected one verified CLS cache for {mode}/{seed}/{split}; found {len(candidates)}'
        with np.load(candidates[0], allow_pickle=False) as cache:
            x, ids = cache['x'].copy(), cache['sample_ids'].copy()
        assert np.array_equal(ids, expected_ids), 'Cache sample ordering mismatch'
        assert x.ndim == 2 and x.shape == (len(ids), 192) and np.isfinite(x).all()
        return x.astype(np.float32), checkpoint_hash

    class ReplicationSAE(nn.Module):
        def __init__(self,width,dictionary,k):
            super().__init__(); self.k=k
            self.pre_bias=nn.Parameter(torch.zeros(width))
            self.encoder=nn.Linear(width,dictionary)
            self.decoder=nn.Linear(dictionary,width,bias=False)
            self.register_buffer('inactive_samples',torch.zeros(dictionary,dtype=torch.long))
        @staticmethod
        def sparsify(values,k):
            v,i=values.topk(k,dim=-1)
            return torch.zeros_like(values).scatter_(-1,i,v)
        def forward(self,x):
            raw=F.relu(self.encoder(x-self.pre_bias))
            acts=self.sparsify(raw,self.k)
            return self.decoder(acts)+self.pre_bias,acts,raw
        def auxiliary_loss(self,x,recon,acts,raw):
            with torch.no_grad():
                self.inactive_samples.add_(len(x))
                self.inactive_samples[(acts>0).any(0)]=0
            dead=self.inactive_samples>=PLAN['dead_after_samples']
            k=min(PLAN['auxk'],int(dead.sum()))
            if not k:
                return x.new_zeros(())
            candidate=raw.masked_fill(~dead.unsqueeze(0),0.)
            return F.mse_loss(self.decoder(self.sparsify(candidate,k)),(x-recon).detach())

    class CenteredSpatialSAE(nn.Module):
        def __init__(self,sae,offset):
            super().__init__(); self.sae=sae
            self.register_buffer('offset',torch.as_tensor(offset,dtype=torch.float32))
        def forward(self,x):
            recon,acts,raw=self.sae(x-self.offset)
            return recon+self.offset,acts,raw

    def source_encoder(mode,seed,expected_hash):
        # Reuse one frozen encoder at a time to bound GPU memory.
        key=(mode,seed)
        if key not in _encoder_cache:
            _encoder_cache.clear()
            if DEVICE.type=='cuda': torch.cuda.empty_cache()
            assert exp.model_digest(mode,seed)==expected_hash
            encoder=exp.Encoder(CFG,mode).to(DEVICE)
            encoder.load_state_dict(exp.in_memory_models[(mode,seed)])
            encoder.eval()
            for parameter in encoder.parameters(): parameter.requires_grad_(False)
            _encoder_cache[key]=encoder
        return _encoder_cache[key]

    def fit_replication(mode,enc_seed,sae_seed,centered,k,variant,implementation_hash):
        train,checkpoint_hash=cached_cls(mode,enc_seed,'train')
        val,_=cached_cls(mode,enc_seed,'val')
        mean=train.astype(np.float64).mean(0).astype(np.float32)
        offset=mean if centered else np.zeros_like(mean)
        rng=np.random.default_rng(PLAN['sample_seed'])
        indices=np.arange(len(train),dtype=np.int64)  # Full training split; no subset selection.
        folder=REPLICATION_OUT/mode/f'encoder_{enc_seed}'/f'center{int(centered)}_k{k}_{variant}_sae{sae_seed}'
        folder.mkdir(parents=True,exist_ok=True)
        protocol={'plan':PLAN,'source_checkpoint':checkpoint_hash,'source_manifest':CFG['manifest_sha256'],
            'source_runner_sha256':CFG['source_sha256'],'implementation_hash':implementation_hash,
            'mode':mode,'encoder_seed':enc_seed,'sae_seed':sae_seed,'centered':centered,'k':k,'variant':variant,
            'train_indices_sha256':hashlib.sha256(indices.tobytes()).hexdigest(),
            'offset':offset.tolist(),'full_train_mean':mean.tolist()}
        registration=folder/'protocol.json'
        if registration.exists():
            assert json.loads(registration.read_text())==protocol,'Protocol/code changed: choose a new pilot name'
        else:
            write_json(registration,protocol)
        np.save(folder/'training_indices.npy',indices)
        exp.seed_all(sae_seed)
        model=ReplicationSAE(train.shape[1],PLAN['dictionary_size'],k).to(DEVICE)
        optimizer=torch.optim.Adam(model.parameters(),lr=PLAN['lr'])
        history=[]; start=0
        pass
        if history:write_json(folder/'history.json',history)
        tensor=torch.from_numpy(train[indices]-offset)
        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor),batch_size=PLAN['batch_size'],shuffle=True)
        for epoch in range(start,PLAN['epochs']):
            check_budget()
            model.train(); total=aux_total=count=nonzero_batches=0
            union=torch.zeros(PLAN['dictionary_size'],dtype=torch.bool,device=DEVICE)
            gradient_norm=0.; norm_count=0
            for (batch,) in loader:
                batch=batch.to(DEVICE); optimizer.zero_grad(set_to_none=True)
                recon,acts,raw=model(batch)
                loss=F.mse_loss(recon,batch)
                aux=model.auxiliary_loss(batch,recon,acts,raw) if variant=='auxk' else batch.new_zeros(())
                objective=loss+PLAN['aux_weight']*aux
                assert torch.isfinite(objective),'Nonfinite objective'
                objective.backward()
                dead=model.inactive_samples>=PLAN['dead_after_samples']
                if variant=='auxk' and dead.any():
                    gradient_norm+=float(model.encoder.weight.grad[dead].norm()); norm_count+=1
                nn.utils.clip_grad_norm_(model.parameters(),PLAN['gradient_clip']); optimizer.step()
                union|=(acts>0).any(0)
                total+=float(loss.detach())*len(batch); aux_total+=float(aux.detach())*len(batch)
                nonzero_batches+=int(float(aux.detach())>0); count+=len(batch)
            record={'epoch':epoch+1,'train_mse':total/count,'aux_mse':aux_total/count,
                'weighted_aux_loss':PLAN['aux_weight']*aux_total/count,'nonzero_aux_batches':nonzero_batches,
                'train_alive_this_epoch':int(union.sum()),
                'mean_dead_encoder_total_gradient_norm':gradient_norm/max(norm_count,1),
                'dead_units':int((model.inactive_samples>=PLAN['dead_after_samples']).sum()) if variant=='auxk' else None,
                'decoder_norm_max':float(model.decoder.weight.detach().norm(dim=0).max())}
            history.append(record)
            pass
            write_json(folder/'history.json',history)
            print(mode,enc_seed,sae_seed,'centered',centered,'K',k,variant,'epoch',epoch+1,'mse',round(total/count,7))
        model.eval()
        # In centered coordinates the training-mean reference is mean-offset.
        acts,reconstruction=evaluate_cls_stream(model,val-offset,mean-offset)
        # Independent check in original feature coordinates, using float64 SSE.
        residual=0.; reference=float(np.sum((val.astype(np.float64)-mean)**2))
        with torch.no_grad():
            for b in range(0,len(val),512):
                predicted=model(torch.from_numpy(val[b:b+512]-offset).to(DEVICE))[0].cpu().numpy()+offset
                residual+=float(np.sum((val[b:b+512].astype(np.float64)-predicted.astype(np.float64))**2))
        original_ev=1-residual/reference
        assert np.isclose(original_ev,reconstruction['variance_explained_train_mean'],atol=2e-5)
        support=acts>0
        labels=MANIFEST.loc[MANIFEST.split=='val','dominant_label'].to_numpy()
        selectivity,_,_=exp.selectivity(acts,labels,CFG['top_m'],CFG['permutations'],sae_seed)
        result={'mode':mode,'encoder_seed':enc_seed,'sae_seed':sae_seed,'centered':centered,'k':k,'variant':variant,
            'epochs':PLAN['epochs'],'train_slices':len(indices),'split':'val',**reconstruction,
            'original_coordinate_EV':original_ev,**selectivity,
            'active_on_every_slice':int(support.all(0).sum()),
            'distinct_support_patterns':int(np.unique(np.packbits(support,axis=1),axis=0).shape[0]),
            'interpretation':'validation replication; average SAE seeds within encoder before across-encoder inference'}

        # Full validation controls; the frozen encoder/probe is unchanged.
        patients=MANIFEST.loc[MANIFEST.split=='val','patient_id'].to_numpy()
        result.update(exp.selectivity_bootstrap(acts,labels,patients,CFG['top_m'],PLAN['selectivity_bootstrap_repeats'],sae_seed))
        result['purity_minus_permutation']=result['mean_purity']-result['permutation_mean_purity']
        _,features,top=exp.selectivity(acts,labels,CFG['top_m'],CFG['permutations'],sae_seed)
        features.to_csv(folder/'validation_features.csv',index=False)
        ordered=features.sort_values(['purity','positive_support','feature'],ascending=[False,False,True])
        selected=int(ordered.iloc[0].feature) if len(ordered) else None
        alive=np.where(support.any(0))[0]
        write_json(folder/'validation_selection.json',{'feature':selected,'spatial_features':alive.tolist(),
            'rule':'validation purity, then positive support, then feature index'})
        if PLAN['spatial_control']:
            encoder=source_encoder(mode,enc_seed,checkpoint_hash)
            adjusted=CenteredSpatialSAE(model,offset).to(DEVICE)
            spatial_table,spatial_protocol=exp.spatial_audit(CFG,encoder,adjusted,VAL_FRAME,alive,'cls')
            spatial_protocol.update({'centering':'training CLS mean subtracted from each patch token',
                'scope':'CLS-to-patch transfer control, not a patch-trained segmentation model'})
            spatial_table.to_csv(folder/'validation_spatial_all_features.csv',index=False)
            write_json(folder/'validation_spatial_protocol.json',spatial_protocol)
            result['selected_feature_dice']=float(spatial_table.loc[spatial_table.feature==selected,
                'mean_dice_all_sampled_slices'].iloc[0]) if selected is not None else None
            result['mean_feature_dice']=float(spatial_table.mean_dice_all_sampled_slices.mean()) if len(spatial_table) else None
            result['spatial_scope']='CLS-to-patch transfer using CLS training mean'
        write_json(folder/'validation_metrics.json',result)
        return result
    ''').lstrip('\n')
    RUNTIME_SOURCE = _textwrap.dedent(r'''
    import gc, hashlib, importlib.metadata, io, itertools, json, os, pickle, struct, sys, time, types
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.nn import functional as F

    class SessionBudgetReached(RuntimeError):
        pass

    class IndexedArchive:
        def __init__(self,path):
            self.path=Path(path).expanduser().resolve()
            with open(self.path,'rb') as stream:
                self.header=pickle.load(stream)
                if self.header.get('format')!='brats-preprocessed-indexed-pickle-v1':raise ValueError('Wrong PKL dataset format')
                stream.seek(-16,2)
                offset,magic=struct.unpack('<Q8s',stream.read(16))
                if magic!=b'BRATSPK1':raise ValueError('Incomplete PKL file: missing index footer')
                stream.seek(offset);self.index=pickle.load(stream)
            self.offsets=self.index['offsets']
            self.frame=pd.DataFrame(self.header['metadata']).copy()
            if len(self.frame)!=len(self.offsets):raise ValueError('Metadata/index length mismatch')
            self.frame['file_path']=['pkl:'+str(i) for i in range(len(self.frame))]
            self.frame['sample_id']=self.frame.patient_id.astype(str)+':'+self.frame.slice_idx.astype(str)
            if self.frame.duplicated(['patient_id','slice_idx']).any():raise ValueError('Duplicate patient/slice')
            if self.frame.groupby('patient_id').split.nunique().max()!=1:raise ValueError('Patient crosses splits')
            self._handle=None;self._pid=None
        def item(self,reference):
            i=int(str(reference).removeprefix('pkl:'))
            if not 0<=i<len(self.offsets):raise IndexError(i)
            if self._handle is None or self._pid!=os.getpid():
                self._handle=open(self.path,'rb');self._pid=os.getpid()
            self._handle.seek(self.offsets[i])
            return pickle.load(self._handle)
        def close(self):
            if self._handle is not None:self._handle.close();self._handle=None

    def sha(path):
        h=hashlib.sha256()
        with open(path,'rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
        return h.hexdigest()

    def inspect_archive(path):
        store=IndexedArchive(path)
        counts=store.frame.groupby('split').agg(patients=('patient_id','nunique'),slices=('slice_idx','size'))
        if counts.to_dict('index')!=store.header['counts']:raise ValueError('Dataset counts do not match header')
        for split in ['train','val','test']:
            group=store.frame[store.frame.split==split]
            if group.empty:raise ValueError('Missing split: '+split)
            for record in [group.iloc[0],group.iloc[-1]]:
                obj=store.item(record.file_path)
                if obj['image'].shape!=obj['seg_mask'].shape or obj['image'].shape!=(128,128):raise ValueError('Invalid image/mask shape')
                if not np.isfinite(obj['image']).all():raise ValueError('Nonfinite image')
        store.close()
        return {'counts':counts.to_dict('index'),'bytes':store.path.stat().st_size,
                'payload_sha256':store.index['payload_sha256'],'source_metadata_sha256':store.header['source_metadata_sha256']}


    def download_archive(source_type,source_location,destination):
        """Download to local scratch for fast indexed training, with partial resume."""
        destination=Path(destination).expanduser().resolve()
        if destination.is_file():
            inspect_archive(destination)
            print('Using the existing local PKL; its full checksum is checked before training.',flush=True)
            return str(destination)
        if not str(source_location).strip():raise ValueError('Enter the Drive path or remote URI copied from Molab remote storage.')
        if source_type=='google_drive':
            try:
                from gdrive_fsspec import GoogleDriveFileSystem
            except ImportError as exc:
                raise ImportError('Connect Google Drive using Files > Add remote storage and run its generated setup cell first. Molab provides gdrive-fsspec.') from exc
            fs=GoogleDriveFileSystem(token='cache',auth_kwargs={'use_local_webserver':False},use_listings_cache=False,skip_instance_cache=True)
            remote_path=source_location
        elif source_type=='remote_uri':
            import fsspec
            fs,remote_path=fsspec.core.url_to_fs(source_location)
        else:raise ValueError('Remote source must be google_drive or remote_uri')
        info=fs.info(remote_path)
        expected=info.get('size')
        if expected is None:raise ValueError('Remote provider did not report file size; use its generated download code and then choose Local source.')
        expected=int(expected)
        identity={'source_hash':hashlib.sha256((source_type+'|'+source_location).encode()).hexdigest(),
                  'bytes':expected,'etag':str(info.get('etag',info.get('ETag',''))),
                  'mtime':str(info.get('mtime',info.get('modified','')))}
        destination.parent.mkdir(parents=True,exist_ok=True)
        partial=destination.with_suffix(destination.suffix+'.partial')
        state_path=destination.with_suffix(destination.suffix+'.download.json')
        if partial.exists():
            if not state_path.exists() or json.loads(state_path.read_text())!=identity:
                raise ValueError('Partial download belongs to a changed remote source. Choose a different local target path.')
        else:state_path.write_text(json.dumps(identity,indent=2))
        offset=partial.stat().st_size if partial.exists() else 0
        if offset>expected:raise ValueError('Partial file is larger than remote source')
        import shutil
        required=expected-offset
        if shutil.disk_usage(destination.parent).free<required+256*1024*1024:
            raise RuntimeError(f'Not enough local disk space: need about {required/1e9:.2f} GB more for the PKL.')
        print(f'Downloading {expected/1e9:.2f} GB; resuming at {offset/1e9:.2f} GB.',flush=True)
        with fs.open(remote_path,'rb') as source:
            if offset:source.seek(offset)
            with open(partial,'ab',buffering=8*1024*1024) as target:
                progress=offset;last_report=offset
                while True:
                    chunk=source.read(8*1024*1024)
                    if not chunk:break
                    target.write(chunk);progress+=len(chunk)
                    if progress-last_report>=128*1024*1024:
                        print(f'Downloaded {progress/1e9:.2f}/{expected/1e9:.2f} GB',flush=True);last_report=progress
        if partial.stat().st_size!=expected:raise ValueError('Download incomplete; click Download/resume again.')
        inspect_archive(partial)
        os.replace(partial,destination)
        print('Download complete. Full SHA256 verification occurs before training.',flush=True)
        return str(destination)




    class Pipeline:
        def __init__(self,dataset_path,output_dir,core_source,sae_source,base_config,expected_sha,
                     max_hours=10.5,remote_uri='',verify_checksum=True,require_cuda=True):
            self.started=time.monotonic();self.max_seconds=float(max_hours)*3600
            self.root=Path(output_dir).expanduser().resolve();self.root.mkdir(parents=True,exist_ok=True)
            if (self.root/'protocol.json').exists():raise ValueError('No checkpoints: choose a fresh results folder for a new run.')
            self.store=IndexedArchive(dataset_path)
            self.df=self.store.frame.copy()
            self.check_budget()
            if require_cuda and not torch.cuda.is_available():raise RuntimeError('Attach a CUDA GPU in Molab Specs before starting.')
            counts=self.df.groupby('split').agg(patients=('patient_id','nunique'),slices=('slice_idx','size')).to_dict('index')
            if counts!=self.store.header['counts']:raise ValueError('Dataset counts changed')
            fingerprint=self.root/'dataset_verification.json'
            signature={'size':self.store.path.stat().st_size,'mtime_ns':self.store.path.stat().st_mtime_ns,
                       'path':str(self.store.path),'expected_sha256':expected_sha}
            verified=False
            if fingerprint.exists():
                old=json.loads(fingerprint.read_text())
                verified=old.get('signature')==signature and old.get('sha256')==expected_sha
            if verify_checksum and not verified:
                print('Verifying the 12 GB input SHA256 once...',flush=True)
                observed=sha(self.store.path)
                if observed!=expected_sha:raise ValueError('PKL checksum mismatch: re-upload the complete file')
                fingerprint.write_text(json.dumps({'signature':signature,'sha256':observed},indent=2))
            self.exp=types.ModuleType('molab_bundled_runner')
            self.exp.__file__=str(self.root/'bundled_runner.py')
            (self.root/'bundled_runner.py').write_text(core_source,encoding='utf-8')
            exec(compile(core_source,self.exp.__file__,'exec'),self.exp.__dict__)
            self.exp.load_item=self.store.item
            self.exp.check_budget=self.check_budget
            self.exp.VERSION='molab-no-checkpoints-1'
            self.exp_source_hash=hashlib.sha256(core_source.encode()).hexdigest()
            cfg=json.loads(json.dumps(base_config))
            cfg.update(root=str(self.root),output='encoders',metadata='pkl-index',version=self.exp.VERSION,
                       source_sha256=self.exp_source_hash,metadata_sha256=self.store.header['source_metadata_sha256'])
            self.manifest=self.df[['patient_id','slice_idx','split','dominant_label','has_tumor','sample_id']].copy()
            cfg['manifest_sha256']=hashlib.sha256(self.manifest.to_csv(index=False).encode()).hexdigest()
            self.cfg=cfg;self.source=self.root/'encoders';self.source.mkdir(exist_ok=True)
            self.sae_source=sae_source
            self.sae_hash=hashlib.sha256(sae_source.encode()).hexdigest()
            frozen={'pipeline_source_sha256':globals().get('RUNTIME_SHA256','development'),'config':{k:v for k,v in cfg.items() if k not in ('root','output','metadata')},
                    'payload_sha256':self.store.index['payload_sha256'],'counts':counts,
                    'SAE_definition_sha256':self.sae_hash,'stage_scope':'training and validation; no held-out-test predictions'}
            registration=self.root/'protocol.json'
            if registration.exists():
                if json.loads(registration.read_text())!=frozen:raise ValueError('Actual experiment configuration/data/code changed. Choose a new output directory.')
            else:self.exp.json_write(registration,frozen)
            self.exp.json_write(self.source/'config.json',cfg)
            self.manifest.to_csv(self.source/'split_manifest.csv',index=False)
            self.exp.json_write(self.root/'environment.json',{'python':sys.version,'torch':torch.__version__,
                 'cuda':torch.version.cuda,'device':str(self.exp.device()),
                 'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 'packages':{name:importlib.metadata.version(name) for name in ['numpy','pandas','torchvision','timm','scikit-learn']}})
            self.exp.json_write(self.root/'class_frequencies.json',{s:{'patients':int(g.patient_id.nunique()),
                'slices':len(g),'dominant_labels':{str(k):int(v) for k,v in g.dominant_label.value_counts().items()}}
                for s,g in self.df.groupby('split')})
            self.exp.in_memory_models={}
            self._model_hashes={}
            def retain_best_model(mode,seed,model):
                state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
                self.exp.in_memory_models[(mode,seed)]=state
                h=hashlib.sha256()
                for name,value in sorted(state.items()):
                    h.update(name.encode());h.update(str(value.dtype).encode())
                    h.update(str(tuple(value.shape)).encode());h.update(value.numpy().tobytes())
                self._model_hashes[(mode,seed)]=h.hexdigest()
            self.exp.retain_best_model=retain_best_model
            self.exp.model_digest=lambda mode,seed:self._model_hashes[(mode,seed)]
        def check_budget(self):
            return  # No automatic stop when model state cannot be resumed.
        def train_and_extract(self,mode,seed):
            self.check_budget();run=self.source/mode/f'encoder_{seed}'
            self.exp.train_encoder(self.cfg,self.df,self.source,mode,seed)
            self.check_budget()
            model=self.exp.Encoder(self.cfg,mode).to(self.exp.device())
            model.load_state_dict(self.exp.in_memory_models[(mode,seed)])
            model.eval()
            for parameter in model.parameters():parameter.requires_grad_(False)
            for split in ['train','val']:
                self.check_budget()
                x,frame=self.exp.extraction(self.cfg,model,self.df[self.df.split==split].copy(),run,split,'cls')
                if x.shape!=(len(frame),192) or not np.isfinite(x).all():raise ValueError('Invalid CLS cache')
                print('CLS ready:',mode,seed,split,x.shape,flush=True);del x
            self.exp.json_write(run/'encoder_complete.json',{'mode':mode,'seed':seed,'epochs':self.cfg['encoder_epochs'],
                'encoder_state_sha256':self.exp.model_digest(mode,seed),'manifest_sha256':self.cfg['manifest_sha256'],
                'weights_persisted':False})
            del model;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()

        def sae_context(self,mode):
            plan={'name':'sae_centered_'+mode+'_v1','modes':[mode],'encoder_seeds':[10,42,52],'sae_seeds':[10,42,52],
              'centered':[True],'ks':[24,8],'variants':['baseline'],'dictionary_size':1536,'epochs':30,
              'batch_size':512,'train_slice_limit':None,'sample_seed':123456,'lr':0.0005,'auxk':96,
              'aux_weight':1/32,'dead_after_samples':138399,'gradient_clip':1.0,
              'selectivity_bootstrap_repeats':200,'spatial_control':True,'primary_k':24,'secondary_k':8}
            context={'np':np,'pd':pd,'torch':torch,'nn':nn,'F':F,'json':json,'hashlib':hashlib,'Path':Path,
              'exp':self.exp,'SOURCE':self.source,'CFG':self.cfg,'MANIFEST':self.manifest,
              'VAL_FRAME':self.df[self.df.split=='val'].copy(),'PLAN':plan,
              'DEVICE':self.exp.device(),'REPLICATION_OUT':self.root/plan['name'],'_encoder_cache':{},
              'check_budget':self.check_budget}
            exec(compile(self.sae_source,'bundled_centered_sae','exec'),context)
            return context
        def validation_probe(self,mode,seed):
            self.check_budget();context=self.sae_context(mode)
            train,checkpoint_hash=context['cached_cls'](mode,seed,'train')
            val,_=context['cached_cls'](mode,seed,'val')
            result_path=self.source/mode/f'encoder_{seed}'/'validation_probe.json'
            identity={'checkpoint_sha256':checkpoint_hash,'manifest_sha256':self.cfg['manifest_sha256']}
            if result_path.exists() and json.loads(result_path.read_text()).get('identity')==identity:return
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            probe=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000))
            probe.fit(train,self.df.loc[self.df.split=='train','has_tumor'].to_numpy())
            predictions=probe.predict(val)
            truth=self.df.loc[self.df.split=='val','has_tumor'].to_numpy()
            patients=self.df.loc[self.df.split=='val','patient_id'].to_numpy()
            result=self.exp.patient_bootstrap(predictions==truth,patients,self.cfg['bootstrap_repeats'],seed)
            result.update({'mode':mode,'encoder_seed':seed,'split':'val','identity':identity,
                           'representation':self.exp.representation_diagnostics(val)})
            self.exp.json_write(result_path,result)
            del train,val,context;gc.collect()
        def fit_saes(self,mode):
            context=self.sae_context(mode);plan=context['PLAN']
            for seed in getattr(self,'selected_seeds',plan['encoder_seeds']):
                marker=self.source/mode/f'encoder_{seed}'/'encoder_complete.json'
                if not marker.exists():raise ValueError(f'Finish encoder and embedding stages for {mode}/{seed} first')
            try:
                for seed,sae_seed,k in itertools.product(getattr(self,'selected_seeds',plan['encoder_seeds']),plan['sae_seeds'],plan['ks']):
                    self.check_budget()
                    folder=context['REPLICATION_OUT']/mode/f'encoder_{seed}'/f'center1_k{k}_baseline_sae{sae_seed}'
                    metric=folder/'validation_metrics.json'
                    if metric.exists():
                        registered=json.loads((folder/'protocol.json').read_text())
                        if registered['implementation_hash']!=self.sae_hash or registered['plan']!=plan or registered['source_checkpoint']!=self.exp.model_digest(mode,seed):
                            raise ValueError('Completed SAE protocol mismatch')
                        print('Reusing completed SAE:',mode,seed,sae_seed,k,flush=True);continue
                    context['fit_replication'](mode,seed,sae_seed,True,k,'baseline',self.sae_hash)
                    pass
            finally:
                context['_encoder_cache'].clear();gc.collect()
                if torch.cuda.is_available():torch.cuda.empty_cache()
        def run(self,modes,stage='all'):
            if len(modes)!=1 or modes[0] not in ('ncl','simclr','supervised'):
                raise ValueError('Use one method notebook at a time')
            if stage!='all':raise ValueError('No checkpoints: run encoder and SAEs together in one worker')
            selected=list(getattr(self,'selected_seeds',self.cfg['encoder_seeds']))
            if not selected or set(selected)-set(self.cfg['encoder_seeds']):raise ValueError('Invalid encoder seed selection')
            try:
                for seed in selected:
                    self.selected_seeds=[seed]
                    mode=modes[0]
                    # Embeddings are scratch caches and can be regenerated from the
                    # restored best encoder, including when resuming the SAE stage.
                    if stage in ('all','encoders','saes'):
                        if stage=='saes' and not (self.source/mode/f'encoder_{seed}'/'encoder_complete.json').exists():
                            raise ValueError('Finish this seed encoder first')
                        self.train_and_extract(mode,seed)
                        self.validation_probe(mode,seed)
                    if stage in ('all','saes'):self.fit_saes(mode)
                status='completed selected stages'
            except SessionBudgetReached as exc:
                status=str(exc);print(status,flush=True)
            finally:
                self.selected_seeds=selected
                try:pass
                finally:self.store.close()
            report={'status':status,'methods':list(modes),'encoder_seeds':selected,'stage':stage,
                    'output_dir':str(self.root),'elapsed_hours':(time.monotonic()-self.started)/3600,'test_evaluated':False}
            self.exp.json_write(self.root/'last_session.json',report)
            return report


    def results_tables(root):
        root=Path(root)
        encoders=[]
        for mode in ['ncl','simclr','supervised']:
            for seed in [10,42,52]:
                p=root/'encoders'/mode/f'encoder_{seed}'/'history.json'
                h=json.loads(p.read_text()) if p.exists() else []
                encoders.append({'mode':mode,'encoder_seed':seed,'epochs':len(h),'target':50,
                                 'best_validation_loss':min((r['val_loss'] for r in h),default=None)})
        metrics=[json.loads(p.read_text()) for p in root.glob('sae_centered_*_v1/*/encoder_*/*/validation_metrics.json')]
        runs=pd.DataFrame(metrics)
        if runs.empty:return pd.DataFrame(encoders),runs,pd.DataFrame()
        measures=[c for c in ['variance_explained_train_mean','alive_features','eligible_features','mean_purity',
                 'purity_minus_permutation','mean_entropy_nats','mean_feature_dice','selected_feature_dice'] if c in runs]
        per_encoder=runs.groupby(['mode','k','encoder_seed'])[measures].mean().reset_index()
        summary=per_encoder.groupby(['mode','k'])[measures].agg(['mean','std','count'])
        runs.to_csv(root/'validation_all_sae_runs.csv',index=False)
        per_encoder.to_csv(root/'validation_per_encoder.csv',index=False)
        summary.to_csv(root/'validation_summary.csv')
        return pd.DataFrame(encoders),runs,summary
    ''').lstrip('\n')
    WORKER_SOURCE = _textwrap.dedent(r'''
    import json,os,sys,time,traceback,types,hashlib
    from pathlib import Path

    request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    root=Path(request['output_dir']).resolve()
    root.mkdir(parents=True,exist_ok=True)
    state_file=root/'worker_state.json'
    lock_file=open(root/'training_worker.lock','a+b')
    if lock_file.tell()==0:lock_file.write(b'0');lock_file.flush()
    lock_file.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(lock_file.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock_file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except (OSError,BlockingIOError):
        print('Another worker holds the training lock; duplicate launch refused.',flush=True)
        sys.exit(2)

    def write_state(status,**extra):
        value={'status':status,'pid':os.getpid(),'output_dir':str(root),'time':time.time(),**extra}
        tmp=state_file.with_name('worker_state.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,state_file)

    try:
        write_state('initializing')
        engine=types.ModuleType('molab_worker_runtime')
        exec(compile(request['runtime_source'],'bundled_runtime','exec'),engine.__dict__)
        engine.RUNTIME_SHA256=hashlib.sha256(request['runtime_source'].encode()).hexdigest()
        pipeline=engine.Pipeline(dataset_path=request['dataset_path'],output_dir=str(root),
             core_source=request['core_source'],sae_source=request['sae_source'],base_config=request['base_config'],
             expected_sha=request['expected_sha'],max_hours=request['max_hours'],
             remote_uri=request['remote_uri'],verify_checksum=request['verify_checksum'])
        pipeline.selected_seeds=request['encoder_seeds']
        original_budget=pipeline.check_budget
        def check_pause_and_budget():
            if (root/'pause_after_epoch.flag').exists():
                raise engine.SessionBudgetReached('Stop requested. Model state is not saved; a new run must retrain.')
            original_budget()
        pipeline.check_budget=check_pause_and_budget
        pipeline.exp.check_budget=check_pause_and_budget
        write_state('running',methods=request['methods'],stage=request['stage'])
        report=pipeline.run(request['methods'],request['stage'])
        engine.results_tables(root)
        write_state(report['status'],report=report)
    except BaseException as exc:
        write_state('error',error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        raise
    finally:
        lock_file.close()
    ''').lstrip('\n')
    CONTROLLER_SOURCE = _textwrap.dedent(r'''
    import json,os,sys,subprocess,time
    from pathlib import Path
    import pandas as pd
    import psutil

    def atomic_json(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(tmp,path)

    def worker_status(root):
        root=Path(root).expanduser().resolve();p=root/'worker_state.json'
        info=json.loads(p.read_text()) if p.exists() else {'status':'not started','output_dir':str(root)}
        alive=False
        if info.get('pid'):
            try:
                process=psutil.Process(int(info['pid']))
                alive=process.is_running() and process.status()!=psutil.STATUS_ZOMBIE and any('molab_training_worker.py' in str(a) for a in process.cmdline())
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        info['alive']=alive
        if not alive and info['status'] in ['starting','initializing','running']:
            info['status']='worker exited; inspect worker.log; no model resume available'
        return info

    def launch_job(applied,dataset_path,core_source,sae_source,runtime_source,base_config,expected_sha,worker_source):
        root=Path(applied['output_dir']).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
        previous=worker_status(root)
        if previous['alive']:return {**previous,'launch':'already running; no duplicate started'}
        if (root/'protocol.json').exists():raise RuntimeError('This run cannot resume. Choose a fresh results folder to retrain.')
        # Do not use this launcher while the older synchronous notebook cell runs.
        worker=root/'molab_training_worker.py';worker.write_text(worker_source,encoding='utf-8')
        request=root/'worker_request.json'
        atomic_json(request,{'output_dir':str(root),'dataset_path':str(Path(dataset_path).expanduser().resolve()),
            'core_source':core_source,'sae_source':sae_source,'runtime_source':runtime_source,
            'base_config':base_config,'expected_sha':expected_sha,'max_hours':applied['max_hours'],
            'remote_uri':applied['remote_uri'],'verify_checksum':applied['verify_checksum'],
            'methods':list(applied['methods']),'encoder_seeds':applied['encoder_seeds'],'stage':applied['stage']})
        (root/'pause_after_epoch.flag').unlink(missing_ok=True)
        kwargs={'start_new_session':True} if os.name!='nt' else {'creationflags':subprocess.CREATE_NO_WINDOW}
        with open(root/'worker.log','ab',buffering=0) as log:
            process=subprocess.Popen([sys.executable,'-u',str(worker),str(request)],stdout=log,stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL,cwd=str(root),**kwargs)
        # Avoid overwriting a more recent state published by an already-started worker.
        if not (root/'worker_state.json').exists() or json.loads((root/'worker_state.json').read_text()).get('pid')!=process.pid:
            atomic_json(root/'worker_state.json',{'pid':process.pid,'status':'starting','time':time.time(),'output_dir':str(root)})
        return {'pid':process.pid,'status':'background worker launched','output_dir':str(root)}

    def request_pause(root):
        root=Path(root).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
        (root/'pause_after_epoch.flag').write_text('Stop at the next epoch/stage boundary. Model state is not saved.')
        return 'Stop requested. Wait for the worker to stop, then download CSVs. Training cannot resume.'

    def read_snapshot(root):
        root=Path(root).expanduser().resolve()
        encoders=[]
        for mode in ['ncl','simclr','supervised']:
            for seed in [10,42,52]:
                p=root/'encoders'/mode/f'encoder_{seed}'/'history.json'
                h=json.loads(p.read_text()) if p.exists() else []
                encoders.append({'mode':mode,'encoder_seed':seed,'epochs':len(h),'target':50,
                    'best_validation_loss':min((r['val_loss'] for r in h),default=None),
                    'history_file_found':p.exists()})
        runs=pd.DataFrame([json.loads(p.read_text()) for p in root.glob('sae_centered_*_v1/*/encoder_*/*/validation_metrics.json')])
        summary=pd.DataFrame()
        if len(runs):
            measures=[c for c in ['variance_explained_train_mean','alive_features','eligible_features','mean_purity',
                  'purity_minus_permutation','mean_entropy_nats','mean_feature_dice','selected_feature_dice'] if c in runs]
            per_encoder=runs.groupby(['mode','k','encoder_seed'])[measures].mean().reset_index()
            summary=per_encoder.groupby(['mode','k'])[measures].agg(['mean','std','count']).reset_index()
            summary.columns=['_'.join(str(v) for v in c if str(v)) if isinstance(c,tuple) else str(c) for c in summary.columns]
        log=root/'worker.log'
        tail=''
        if log.exists():
            with open(log,'rb') as stream:
                stream.seek(max(0,log.stat().st_size-18000))
                tail=stream.read().decode('utf-8',errors='replace')
        return worker_status(root),pd.DataFrame(encoders),runs,summary,tail

    def csv_download(root,method,encoder_seed=None):
        import io,zipfile
        root=Path(root).expanduser().resolve()
        state,enc,runs,summary,tail=read_snapshot(root)
        enc=enc[enc['mode']==method]
        if encoder_seed is not None:
            enc=enc[enc.encoder_seed==encoder_seed]
            if len(runs):runs=runs[(runs['mode']==method)&(runs.encoder_seed==encoder_seed)]
        histories=[];probes=[]
        for seed in ([encoder_seed] if encoder_seed is not None else [10,42,52]):
            folder=root/'encoders'/method/f'encoder_{seed}'
            p=folder/'history.json'
            if p.exists():
                histories.extend(dict(mode=method,encoder_seed=seed,**row) for row in json.loads(p.read_text()))
            p=folder/'validation_probe.json'
            if p.exists():
                probes.append(json.loads(p.read_text()))
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w',zipfile.ZIP_DEFLATED) as archive:
            for name,frame in [('encoder_progress',enc),('encoder_history',pd.DataFrame(histories)),
                               ('validation_probe',pd.json_normalize(probes)),('sae_runs',runs),('sae_summary',summary)]:
                archive.writestr(method+'_'+name+'.csv',frame.to_csv(index=False))
            archive.writestr('README.txt','CSV exports are for inspection. No model checkpoints are saved. Training cannot resume; model weights exist only in RAM during the worker run.')
        return data.getvalue()
    ''').lstrip('\n')
    BASE_CONFIG = {'experiment_name': 'molab_drive_checkpointed_v2', 'encoder_seeds': [10, 42, 52], 'sae_seeds': [10, 42, 52], 'modes': ['ncl'], 'metadata': 'data/processed_flair_brats2023_expansion_v1/metadata.csv', 'output': 'results/brats2023_expansion_v1', 'img_size': 128, 'batch_size': 64, 'encoder_epochs': 50, 'encoder_lr': 0.0005, 'weight_decay': 0.01, 'temperature': 0.5, 'sae_epochs': 30, 'sae_batch_size': 512, 'sae_lr': 0.0005, 'dictionary_size': 1536, 'sae_k': 24, 'sae_variants': ['baseline'], 'token_spaces': ['cls'], 'train_patch_slice_limit': 2048, 'auxk': 96, 'aux_weight': 0.03125, 'dead_after_samples': 138399, 'top_m': 10, 'permutations': 100, 'bootstrap_repeats': 2000, 'selectivity_bootstrap_repeats': 200, 'spatial_samples': 128, 'spatial_percentile': 75, 'tumor_patch_fraction': 0.1, 'num_workers': 0}
    DATASET_SHA256 = '764780cdc36e126eb94a14ad87e9284756321798d169abd8515a737813154fc3'
    DRIVE_IMPORT_SOURCE = _textwrap.dedent(r'''
    """Embedded in the nine seed notebooks; downloading requires an explicit click."""
    from pathlib import Path
    import hashlib
    import json
    import os

    DRIVE_DATASET_URL = ''
    DRIVE_DATASET_SHA256 = '764780cdc36e126eb94a14ad87e9284756321798d169abd8515a737813154fc3'
    DRIVE_DATASET_BYTES = 12167455448
    HF_DATASET_REPO = 'YOUR_USERNAME/YOUR_PRIVATE_DATASET'


    def import_hf_dataset(destination='data/brats2023_preprocessed_complete.pkl',
                          repo_id=HF_DATASET_REPO, filename='brats2023_preprocessed_complete.pkl',
                          expected_sha256=DRIVE_DATASET_SHA256, expected_bytes=DRIVE_DATASET_BYTES):
        """Download the private HF dataset using HF_TOKEN; never persist/print the token.

        Download directly into the local data directory to avoid a second 12 GB cache
        copy. Existing verified input is reused independently of the download source.
        """
        target = Path(destination).expanduser().resolve()
        if target.is_file():
            return import_drive_dataset(target, expected_sha256=expected_sha256, expected_bytes=expected_bytes)
        token = os.environ.get('HF_TOKEN', '').strip()
        if not token:
            # Molab remote-storage connections can keep secrets in a local .env.
            from dotenv import dotenv_values
            token = str(dotenv_values('.env').get('HF_TOKEN') or '').strip()
        if not token:
            raise RuntimeError('Private dataset: add your read-only Hugging Face token as HF_TOKEN in Molab secrets/environment, then rerun Import / resume dataset. Do not paste the token into notebook code.')
        from huggingface_hub import HfApi, hf_hub_download
        files = HfApi(token=token).list_repo_files(repo_id=repo_id, repo_type='dataset')
        selected = filename
        if selected not in files:
            candidates = [name for name in files if name.lower().endswith('.pkl')]
            matching = [name for name in candidates if Path(name).name == Path(filename).name]
            if len(matching) == 1:
                selected = matching[0]
            elif len(candidates) == 1:
                selected = candidates[0]
            else:
                raise ValueError('Cannot identify one PKL. Set HF filename to the path in Files and versions. Available PKLs: ' + repr(candidates))
        target.parent.mkdir(parents=True, exist_ok=True)
        print('Downloading/resuming private Hugging Face dataset:', repo_id, selected, flush=True)
        downloaded = Path(hf_hub_download(repo_id=repo_id, repo_type='dataset', filename=selected,
            token=token, local_dir=str(target.parent))).resolve()
        if not downloaded.is_file():
            raise RuntimeError('Hugging Face did not return a local dataset file.')
        # Reuse the common full size/SHA256 verifier before handing data to training.
        import_drive_dataset(downloaded, expected_sha256=expected_sha256, expected_bytes=expected_bytes)
        if downloaded != target:
            os.replace(downloaded, target)
            # A renamed file gets a new verification receipt on first subsequent use.
        return str(target)


    def import_drive_dataset(destination='data/brats2023_preprocessed_complete.pkl',
                             url=DRIVE_DATASET_URL, expected_sha256=DRIVE_DATASET_SHA256,
                             expected_bytes=DRIVE_DATASET_BYTES):
        """Download/resume from the shared Drive link and verify before use."""
        target = Path(destination).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        receipt = target.with_suffix(target.suffix + '.verified.json')

        def fingerprint(path):
            stat = path.stat()
            return {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                    'ctime_ns': stat.st_ctime_ns, 'sha256': expected_sha256}

        def verify(path):
            if expected_bytes is not None and path.stat().st_size != expected_bytes:
                raise ValueError('Dataset size mismatch; file retained for inspection: ' + str(path))
            print('Verifying dataset SHA256 (streaming, not loading 12 GB into RAM)...', flush=True)
            h = hashlib.sha256()
            with open(path, 'rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != expected_sha256:
                raise ValueError('Dataset checksum mismatch. Do not train from this file: ' + str(path))

        if target.is_file():
            try:
                cached = json.loads(receipt.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                cached = None
            if cached != fingerprint(target):
                verify(target)
        else:
            try:
                import gdown
            except ImportError as exc:
                raise ImportError('Install gdown>=5.2,<6 in this notebook environment, then rerun the download cell.') from exc
            pending = target.with_suffix(target.suffix + '.gdown')
            # gdown manages its own resumable temporary file. Do not mix this with
            # the authenticated fsspec downloader's .partial file.
            print('Downloading/resuming your prepared PKL from Google Drive...', flush=True)
            result = gdown.download(url=url, output=str(pending), fuzzy=True, resume=True, quiet=False)
            if not result or not pending.is_file():
                raise RuntimeError('Drive download incomplete. Rerun to resume; if quota-limited, select authenticated_drive in settings.')
            verify(pending)
            os.replace(pending, target)
        temp = receipt.with_name(receipt.name + '.' + str(os.getpid()) + '.tmp')
        temp.write_text(json.dumps(fingerprint(target), indent=2), encoding='utf-8')
        os.replace(temp, receipt)
        print('Dataset ready:', target, flush=True)
        return str(target)
    ''').lstrip('\n')
    return METHOD, ENCODER_SEED, CORE_SOURCE, SAE_SOURCE, RUNTIME_SOURCE, WORKER_SOURCE, CONTROLLER_SOURCE, BASE_CONFIG, DATASET_SHA256, DRIVE_IMPORT_SOURCE

@app.cell(hide_code=True)
def _(CONTROLLER_SOURCE, DRIVE_IMPORT_SOURCE, RUNTIME_SOURCE):
    import types as _types
    import hashlib as _hashlib
    engine = _types.ModuleType("no_checkpoint_runtime")
    exec(compile(RUNTIME_SOURCE,"bundled_runtime","exec"),engine.__dict__)
    engine.RUNTIME_SHA256 = _hashlib.sha256(RUNTIME_SOURCE.encode()).hexdigest()
    control = _types.ModuleType("no_checkpoint_control")
    exec(compile(CONTROLLER_SOURCE,"bundled_control","exec"),control.__dict__)
    drive_data = _types.ModuleType("drive_dataset_import")
    exec(compile(DRIVE_IMPORT_SOURCE,"drive_dataset_import","exec"),drive_data.__dict__)
    return control, drive_data, engine

@app.cell
def _(ENCODER_SEED, METHOD, mo):
    import datetime as _datetime
    _suffix = _datetime.datetime.now(_datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
    settings = mo.ui.dictionary({
        "data_source":mo.ui.dropdown(options=["huggingface","drive_link","authenticated_drive","local"],value="huggingface",label="Dataset source"),
        "hf_repo":mo.ui.text(value="YOUR_USERNAME/YOUR_PRIVATE_DATASET",label="Private Hugging Face dataset repo"),
        "hf_filename":mo.ui.text(value="brats2023_preprocessed_complete.pkl",label="HF filename (auto-detects a single PKL if different)"),
        "drive_link":mo.ui.text(value="",label="Dataset Drive link"),
        "source_location":mo.ui.text(value="brats2023_preprocessed_complete.pkl",label="Authenticated Drive PKL path (fallback only)"),
        "dataset_path":mo.ui.text(value="data/brats2023_preprocessed_complete.pkl",label="Local PKL cache"),
        "output_dir":mo.ui.text(value=f"brats_no_checkpoints_{METHOD}_seed{ENCODER_SEED}_{_suffix}",label="Fresh local results folder"),
    }).form(submit_button_label="Apply settings")
    settings
    return (settings,)

@app.cell
def _(ENCODER_SEED, METHOD, mo, settings):
    mo.stop(settings.value is None,mo.md("Apply settings first."))
    applied = dict(settings.value)
    applied.update(methods=[METHOD],encoder_seeds=[ENCODER_SEED],stage='all',max_hours=0,remote_uri='',verify_checksum=True)
    mo.md("**Checkpoint saving: OFF.** Results folder: `"+applied['output_dir']+"`.")
    return (applied,)

@app.cell
def _(mo):
    fetch_dataset = mo.ui.run_button(label="Import / resume dataset")
    fetch_dataset
    return (fetch_dataset,)

@app.cell
def _(applied, drive_data, engine, fetch_dataset, mo):
    from pathlib import Path as _DatasetPath
    _target = _DatasetPath(applied['dataset_path']).expanduser().resolve()
    if not _target.is_file():
        mo.stop(applied['data_source']=='local',mo.md("Local PKL missing; choose Hugging Face or correct the path."))
        mo.stop(not fetch_dataset.value,mo.md("Click Import / resume dataset."))
        if applied['data_source']=='huggingface':
            drive_data.import_hf_dataset(_target,repo_id=applied['hf_repo'],filename=applied['hf_filename'])
        elif applied['data_source']=='drive_link':
            drive_data.import_drive_dataset(_target,url=applied['drive_link'])
        else:
            engine.download_archive('google_drive',applied['source_location'],_target)
    dataset_ready_path = str(_target)
    _inspection = engine.inspect_archive(_target)
    mo.ui.table([dict(split=_s,**_v) for _s,_v in _inspection['counts'].items()],selection=None)
    return (dataset_ready_path,)

@app.cell
def _(mo):
    start_run = mo.ui.run_button(label="Start training")
    stop_run = mo.ui.run_button(label="Stop after epoch (cannot resume)")
    mo.hstack([start_run,stop_run])
    return start_run, stop_run

@app.cell
def _(BASE_CONFIG, CORE_SOURCE, DATASET_SHA256, RUNTIME_SOURCE, SAE_SOURCE, WORKER_SOURCE,
      applied, control, dataset_ready_path, mo, start_run):
    mo.stop(not start_run.value)
    launch_status = control.launch_job(applied,dataset_ready_path,CORE_SOURCE,SAE_SOURCE,
        RUNTIME_SOURCE,BASE_CONFIG,DATASET_SHA256,WORKER_SOURCE)
    mo.md(str(launch_status))
    return (launch_status,)

@app.cell
def _(applied, control, mo, stop_run):
    mo.stop(not stop_run.value)
    mo.md(control.request_pause(applied['output_dir']))
    return

@app.cell
def _(mo):
    refresh_progress = mo.ui.refresh(default_interval="10s",label="Live progress")
    refresh_progress
    return (refresh_progress,)

@app.cell
def _(ENCODER_SEED, METHOD, applied, control, mo, refresh_progress):
    _ = refresh_progress.value
    _state,_enc,_runs,_summary,_tail = control.read_snapshot(applied['output_dir'])
    _enc = _enc[(_enc['mode']==METHOD)&(_enc.encoder_seed==ENCODER_SEED)]
    if len(_runs):_runs=_runs[(_runs['mode']==METHOD)&(_runs.encoder_seed==ENCODER_SEED)]
    _items = [mo.md("**Worker:** "+str(_state)),mo.md("**No model checkpoints or Drive backup.**"),
        mo.ui.table(_enc,selection=None),mo.md(f"Completed SAE fits: **{len(_runs)} / 6**.")]
    if len(_summary):_items.append(mo.ui.table(_summary,selection=None))
    _items += [mo.download(data=control.csv_download(applied['output_dir'],METHOD,ENCODER_SEED),
        filename=METHOD+'_seed'+str(ENCODER_SEED)+'_results.zip',mimetype='application/zip',label='Download CSV results'),
        mo.md("```text\n"+_tail+"\n```")]
    mo.vstack(_items)
    return

if __name__ == "__main__":
    app.run()
