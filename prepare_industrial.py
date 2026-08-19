"""
Prepare sliding-window train / val / test splits for the US Industrial Landscape dataset.

Input  : US_Industrial_Landscape_complete_tensor.npz   shape (N=10, F=10, T=189)
Output : data/industrial_L{seq_in}_H{seq_out}/
           train.npz   x=(S_tr, L, N, F) normalized,  x_raw same in original units,
                       y=(S_tr, H, N, F) raw
           val.npz     x=(S_va, L, N, F) normalized,  x_raw, y=(S_va, H, N, F) raw
           test.npz    x=(S_te, L, N, F) normalized,  x_raw, y=(S_te, H, N, F) raw
           scaler.npz           mean=(N, F)  std=(N, F)   fitted on train X only
           scaler_trainval.npz  mean=(N, F)  std=(N, F)   fitted on train + val X
           meta.json   split sizes, date ranges, node/feature names

Normalization: per-node x per-feature z-score.
  Each (node, feature) pair is independently centred and scaled so that
  sectors with very different absolute magnitudes (e.g. Mining ~600 K vs
  Trade ~25 000 K employment) contribute equally to the loss.

  Two scalers are written because the scaler has to be fitted on whatever the
  model is actually trained on.  Hyper-parameter selection trains on `train`
  and therefore uses scaler.npz; the final refit trains on train+val and uses
  scaler_trainval.npz.  The distinction matters on this dataset: the COVID
  shock falls inside the validation window, so the train-only scaler leaves
  test inputs at |z| up to 131.

Usage:
    python prepare_industrial.py                         # default L=12, H=24
    python prepare_industrial.py --seq_in 12 --seq_out 24
"""
import argparse
import json
from pathlib import Path

import numpy as np

from data_utils import fit_scaler

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--npz",         type=str,   default="US_Industrial_Landscape_complete_tensor.npz")
parser.add_argument("--seq_in",      type=int,   default=12, help="lookback L in months")
parser.add_argument("--seq_out",     type=int,   default=24, help="max forecast horizon H in months")
parser.add_argument("--train_ratio", type=float, default=0.70)
parser.add_argument("--val_ratio",   type=float, default=0.15)
parser.add_argument("--out_dir",     type=str,   default="",
                    help="output directory (default: data/industrial_L{seq_in}_H{seq_out})")
args = parser.parse_args()

L, H = args.seq_in, args.seq_out
out_dir = Path(args.out_dir) if args.out_dir else Path(f"data/industrial_L{L}_H{H}")
out_dir.mkdir(parents=True, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
d = np.load(args.npz)
data          = d["data"].astype(np.float32)   # (N, F, T)
dates         = d["dates"]                      # (T,)  e.g. '2010-01-01'
node_ids      = d["node_ids"]
node_names    = d["node_names"]
feature_names = d["feature_names"]

N, F, T = data.shape
print(f"Loaded  : N={N} nodes, F={F} features, T={T} months")
print(f"Period  : {dates[0]} -> {dates[-1]}")
print(f"L={L}  H={H}")

# ── Sliding windows ────────────────────────────────────────────────────────────
data_tnf = data.transpose(2, 0, 1)      # (T, N, F)

n_samples = T - L - H + 1
if n_samples <= 0:
    raise ValueError(f"Not enough data: T={T}, L+H={L+H} requires T > {L+H-1}")

X = np.stack([data_tnf[i      : i + L    ] for i in range(n_samples)])  # (S, L, N, F)
Y = np.stack([data_tnf[i + L  : i + L + H] for i in range(n_samples)])  # (S, H, N, F)

dates_x_end = np.array([dates[i + L - 1]     for i in range(n_samples)])  # last input date
dates_y_end = np.array([dates[i + L + H - 1] for i in range(n_samples)])  # last target date

# ── Train / val / test split (time-ordered, no shuffle) ───────────────────────
n_train = int(n_samples * args.train_ratio)
n_val   = int(n_samples * args.val_ratio)
n_test  = n_samples - n_train - n_val

print(f"\nSamples : train={n_train}, val={n_val}, test={n_test}  (total={n_samples})")
print(f"  Train targets end at : {dates_y_end[n_train - 1]}")
print(f"  Val   targets end at : {dates_y_end[n_train + n_val - 1]}")
print(f"  Test  targets end at : {dates_y_end[-1]}")

# ── Per-node × per-feature normalisation ──────────────────────────────────────
# X shape: (S, L, N, F)  ->  average over axes 0,1 (samples & time) → (N, F)
mean,    std    = fit_scaler(X[:n_train])
mean_tv, std_tv = fit_scaler(X[:n_train + n_val])

def normalize(arr):
    """arr shape (S, L, N, F) — broadcast (N, F) over leading axes."""
    return (arr - mean) / std

# X is normalised; Y stays in original scale (inverse-transform happens in training)
splits = [
    ("train", X[:n_train],             Y[:n_train]),
    ("val",   X[n_train:n_train+n_val], Y[n_train:n_train+n_val]),
    ("test",  X[n_train+n_val:],        Y[n_train+n_val:]),
]

for name, xp, yp in splits:
    np.savez_compressed(
        out_dir / f"{name}.npz",
        x=normalize(xp),   # (S, L, N, F)  normalised with the train-only scaler
        x_raw=xp,          # (S, L, N, F)  original units, re-normalisable
        y=yp,              # (S, H, N, F)  raw original scale
    )

np.savez_compressed(out_dir / "scaler.npz",          mean=mean,    std=std)
np.savez_compressed(out_dir / "scaler_trainval.npz", mean=mean_tv, std=std_tv)

x_te = X[n_train + n_val:]
print(f"\nTest-input extremes under each scaler (max |z|):")
print(f"  train only : {np.abs((x_te - mean)    / std   ).max():8.1f}")
print(f"  train+val  : {np.abs((x_te - mean_tv) / std_tv).max():8.1f}")

# ── Metadata ───────────────────────────────────────────────────────────────────
meta = {
    "seq_in":       L,
    "seq_out":      H,
    "n_nodes":      N,
    "n_features":   F,
    "n_train":      n_train,
    "n_val":        n_val,
    "n_test":       n_test,
    "train_ratio":  args.train_ratio,
    "val_ratio":    args.val_ratio,
    "node_ids":     [str(x) for x in node_ids],
    "node_names":   [str(x) for x in node_names],
    "feature_names":[str(x) for x in feature_names],
    "scaler_mean":  mean.tolist(),
    "scaler_std":   std.tolist(),
    "scaler_trainval_mean": mean_tv.tolist(),
    "scaler_trainval_std":  std_tv.tolist(),
}
(out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

print(f"\nScaler (mean / std  per node x per feature):")
print(f"  {'Node':<40s}  {'Feature':<50s}  {'mean':>12}  {'std':>10}")
print("  " + "-" * 118)
for ni, nn in enumerate(node_names):
    for fi, fn in enumerate(feature_names):
        print(f"  {str(nn):<40s}  {str(fn):<50s}  {mean[ni, fi]:>12.4f}  {std[ni, fi]:>10.4f}")

print(f"\nOutput : {out_dir}")
print("Done.")
