"""
data_utils.py — Split and scaler loading shared by the training / evaluation scripts.

Two normalisation regimes are supported, selected with `fit_on`:

  "train"     mean/std fitted on the training inputs only.  Used while
              hyper-parameters — including the epoch count — are selected
              against the validation split.

  "trainval"  mean/std fitted on train + validation inputs.  Used for the
              final refit, where train+val *is* the training set.

The scaler must be fitted on whatever the model is actually trained on.  On
this dataset the COVID shock falls inside the validation window, so a
train-only scaler leaves test inputs at |z| up to 131 whereas a train+val
scaler keeps them below 12.

Splits produced by older runs of prepare_industrial.py stored only the
train-normalised inputs.  Those directories still load: the raw inputs are
recovered by inverting the train scaler, and the train+val scaler is derived
and cached on first use.
"""
from pathlib import Path

import json
import numpy as np

FIT_CHOICES = ("train", "trainval")
_SPLITS_FOR = {"train": ("train",), "trainval": ("train", "val")}
_TRAINVAL_SCALER = "scaler_trainval.npz"


def load_meta(data_dir) -> dict:
    return json.loads((Path(data_dir) / "meta.json").read_text())


def _train_scaler(data_dir):
    d = np.load(Path(data_dir) / "scaler.npz")
    return d["mean"].astype(np.float32), d["std"].astype(np.float32)


def raw_inputs(data_dir, name: str) -> np.ndarray:
    """Sliding-window inputs of one split in original units, shape (S, L, N, F)."""
    z = np.load(Path(data_dir) / f"{name}.npz")
    if "x_raw" in z.files:
        return z["x_raw"].astype(np.float32)
    mean, std = _train_scaler(data_dir)
    return (z["x"].astype(np.float32) * std + mean).astype(np.float32)


def raw_targets(data_dir, name: str) -> np.ndarray:
    """Targets of one split in original units, shape (S, H, N, F)."""
    return np.load(Path(data_dir) / f"{name}.npz")["y"].astype(np.float32)


def fit_scaler(x_raw: np.ndarray):
    """Per-node x per-feature mean/std over the sample and time axes."""
    mean = x_raw.mean(axis=(0, 1))
    std = x_raw.std(axis=(0, 1))
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def load_scaler(data_dir, fit_on: str = "train"):
    """Return (mean, std) of shape (N, F), caching the train+val scaler on disk."""
    if fit_on not in FIT_CHOICES:
        raise ValueError(f"fit_on must be one of {FIT_CHOICES}, got {fit_on!r}")
    data_dir = Path(data_dir)
    if fit_on == "train":
        return _train_scaler(data_dir)

    cached = data_dir / _TRAINVAL_SCALER
    if cached.exists():
        d = np.load(cached)
        return d["mean"].astype(np.float32), d["std"].astype(np.float32)

    x = np.concatenate([raw_inputs(data_dir, n) for n in _SPLITS_FOR["trainval"]])
    mean, std = fit_scaler(x)
    np.savez_compressed(cached, mean=mean, std=std)
    return mean, std


def load_split(data_dir, name: str, fit_on: str = "train"):
    """Return (x_normalised, y_raw) for one split, both float32."""
    mean, std = load_scaler(data_dir, fit_on)
    x = ((raw_inputs(data_dir, name) - mean) / std).astype(np.float32)
    return x, raw_targets(data_dir, name)


def load_fit_split(data_dir, fit_on: str = "train"):
    """Return (x, y) of everything the model is fitted on under this regime."""
    parts = [load_split(data_dir, n, fit_on) for n in _SPLITS_FOR[fit_on]]
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]))


def fit_targets(data_dir, fit_on: str = "train") -> np.ndarray:
    """Raw targets of the fitted splits — the reference for the MASE denominator."""
    return np.concatenate([raw_targets(data_dir, n) for n in _SPLITS_FOR[fit_on]])
