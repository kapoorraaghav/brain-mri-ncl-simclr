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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


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
        item = np.load(row.file_path, allow_pickle=True).item()
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
    latest, best = run / "latest.pt", run / "best.pt"
    seed_all(seed)
    model = Encoder(cfg, mode).to(device())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["encoder_lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device().type == "cuda")
    history, start, best_loss = [], 0, float("inf")
    if latest.exists():
        state = torch.load(latest, map_location=device(), weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_scaler(scaler, state.get("scaler"))
        history, start, best_loss = state["history"], state["epoch"] + 1, state["best_loss"]
        restore_rng(state["rng"])
    loader = DataLoader(Slices(df[df.split == "train"], cfg, mode), batch_size=cfg["batch_size"],
                        shuffle=True, drop_last=True, num_workers=cfg["num_workers"])
    if not len(loader):
        raise ValueError("Training loader is empty")
    for epoch in range(start, cfg["encoder_epochs"]):
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
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch, "best_loss": best_loss, "history": history, "config": cfg,
                 "mode": mode, "encoder_seed": seed, "rng": rng_state()}
        save_checkpoint(latest, state)
        if improved:
            save_checkpoint(best, state)
        json_write(run / "history.json", history)
        print(f"{mode} encoder={seed} epoch={epoch+1} train={np.mean(losses):.4f} val={val_loss:.4f}", flush=True)
    return run


def extraction(cfg, model, frame, run, split, space):
    frame = frame.reset_index(drop=True)
    if split == "train" and space == "patch":
        frame = frame.sample(n=min(len(frame), cfg["train_patch_slice_limit"]), random_state=123456).sort_index()
    identity = {"checkpoint": digest(run / "best.pt"), "manifest": cfg["manifest_sha256"],
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
            item = np.load(row.file_path, allow_pickle=True).item()
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
