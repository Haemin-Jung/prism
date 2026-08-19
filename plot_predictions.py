"""
Visualise MTGNN predictions on the US Industrial Landscape test set.

Single panel — Full 24-month forecast fan from the last test sample.
  Input  : the most recent L=12 months of data available before the test period
  Predict: the next H=24 months
  Compare: model prediction vs actual values

Usage examples:
    python plot_predictions.py                          # defaults
    python plot_predictions.py --node 0 --feature 0    # Mining, employment
    python plot_predictions.py --node 2 --feature 3    # Manufacturing, job openings
    python plot_predictions.py --list                   # print node / feature indices
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless — saves PNG without a display
import matplotlib.pyplot as plt
import numpy as np
import torch

from net import gtnet

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",      type=str, default="checkpoints/industrial/mtgnn_fgraph_s1_L12_H24.pth")
parser.add_argument("--data_dir",  type=str, default="data/industrial_L12_H24")
parser.add_argument("--seq_in",    type=int, default=12)
parser.add_argument("--seq_out",   type=int, default=24)
parser.add_argument("--feature_graph", action="store_true", default=True)
parser.add_argument("--device",    type=str, default="cuda:0")
parser.add_argument("--node",      type=int, default=0,  help="node index 0-9")
parser.add_argument("--feature",   type=int, default=0,  help="feature index 0-9")
parser.add_argument("--n_mc",      type=int, default=30, help="MC-Dropout forward passes")
parser.add_argument("--out",       type=str, default="prediction_plot.png")
parser.add_argument("--list",      action="store_true", help="print node/feature names and exit")
args = parser.parse_args()

if args.device.startswith("cuda") and not torch.cuda.is_available():
    args.device = "cpu"
device = torch.device(args.device)

# ── Load metadata & data ──────────────────────────────────────────────────────
data_dir = Path(args.data_dir)
meta = json.loads((data_dir / "meta.json").read_text())
node_names    = meta["node_names"]
feature_names = meta["feature_names"]

if args.list:
    print("Nodes:")
    for i, n in enumerate(node_names):    print(f"  [{i}] {n}")
    print("\nFeatures:")
    for i, f in enumerate(feature_names): print(f"  [{i}] {f}")
    raise SystemExit(0)

_scaler = data_dir / "scaler_trainval.npz"
if not _scaler.exists():
    _scaler = data_dir / "scaler.npz"
scaler_npz  = np.load(_scaler)
# shape (N, F) — per-node x per-feature; slice [ni, fi] for scalar access
scaler_mean = torch.tensor(scaler_npz["mean"], dtype=torch.float32).to(device)  # (N, F)
scaler_std  = torch.tensor(scaler_npz["std"],  dtype=torch.float32).to(device)  # (N, F)

test_npz = np.load(data_dir / "test.npz")
x_test = torch.tensor(test_npz["x"], dtype=torch.float32)   # (S, L, N, F) normalised
y_test = torch.tensor(test_npz["y"], dtype=torch.float32)   # (S, H, N, F) raw

N, F, L, H = 10, 10, args.seq_in, args.seq_out
OUT_DIM = H * F

# ── Load model ────────────────────────────────────────────────────────────────
model = gtnet(
    gcn_true=True, buildA_true=True, gcn_depth=2,
    num_nodes=N, device=device, predefined_A=None,
    dropout=0.3, subgraph_size=10, node_dim=40, dilation_exponential=1,
    conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128,
    seq_length=L, in_dim=F, out_dim=OUT_DIM, layers=3,
    propalpha=0.05, tanhalpha=3.0, layer_norm_affline=True,
    horizon_decoder="gru_ctx_hemb", n_horizon=H, n_features=F,
    feature_graph=args.feature_graph,
).to(device)
model.load_state_dict(torch.load(args.ckpt, map_location=device))

# ── MC-Dropout inference ──────────────────────────────────────────────────────
# Keep dropout ON during inference to sample from posterior predictive.
model.train()   # activates dropout

ni = args.node      # node index
fi = args.feature   # feature index

print(f"Node    : [{ni}] {node_names[ni]}")
print(f"Feature : [{fi}] {feature_names[fi]}")
print(f"MC runs : {args.n_mc}")

S = len(x_test)   # number of test samples

all_mc_preds = []   # list of (n_mc,) arrays, one per (sample, horizon) pair
# Shape we want: (n_mc, S, H)

with torch.no_grad():
    for _ in range(args.n_mc):
        run_preds = []
        for s in range(S):
            xb = x_test[s:s+1].to(device)           # (1, L, N, F)
            inp = xb.permute(0, 3, 2, 1)             # (1, F, N, L)
            out = model(inp)                          # (1, H*F, N, 1)
            out = out.squeeze(-1).view(1, H, F, N).permute(0, 1, 3, 2)  # (1, H, N, F)
            pred_raw = out * scaler_std + scaler_mean                    # (1, H, N, F)
            run_preds.append(pred_raw[0, :, ni, fi].cpu().numpy())       # (H,)
        all_mc_preds.append(np.stack(run_preds, axis=0))   # (S, H)

mc_preds = np.stack(all_mc_preds, axis=0)   # (n_mc, S, H)

# ── Reconstruct calendar dates for test targets ───────────────────────────────
# The complete tensor holds all dates; test targets start at index n_train+n_val+L.
full_npz   = np.load("US_Industrial_Landscape_complete_tensor.npz")
all_dates  = full_npz["dates"]   # (T,) array of date strings
T_total    = len(all_dates)
n_samples  = T_total - L - H + 1
n_train    = int(n_samples * 0.70)
n_val      = int(n_samples * 0.15)

# test sample s → target months are all_dates[n_train+n_val+s+L : n_train+n_val+s+L+H]
test_start_idx = n_train + n_val   # raw data index of first test sample's start

import matplotlib.dates as mdates
from datetime import datetime

def to_dt(s):
    return datetime.strptime(str(s)[:7], "%Y-%m")

# Calendar dates for each h=1 target (one per test sample)
dates_h1 = [to_dt(all_dates[test_start_idx + s + L]) for s in range(S)]

# Calendar dates for the full H-step target of the LAST test sample
last      = S - 1
dates_full = [to_dt(all_dates[test_start_idx + last + L + h]) for h in range(H)]

# --- rolling 1-step-ahead (h=1 for each test sample) ---
pred_h1_mean = mc_preds[:, :, 0].mean(axis=0)    # (S,)
pred_h1_lo   = np.percentile(mc_preds[:, :, 0],  2.5, axis=0)
pred_h1_hi   = np.percentile(mc_preds[:, :, 0], 97.5, axis=0)
actual_h1    = y_test[:, 0, ni, fi].numpy()      # (S,)

# --- last test sample's full H-step trajectory ---
pred_full_mean = mc_preds[:, last, :].mean(axis=0)   # (H,)
pred_full_lo   = np.percentile(mc_preds[:, last, :],  2.5, axis=0)
pred_full_hi   = np.percentile(mc_preds[:, last, :], 97.5, axis=0)
actual_full    = y_test[last, :, ni, fi].numpy()     # (H,)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _scale_mismatch(a, b, thresh=3.0):
    """Return True when predicted mean is >thresh× larger/smaller than actual mean."""
    mean_a = abs(np.mean(a))
    mean_b = abs(np.mean(b))
    if mean_a < 1e-8 or mean_b < 1e-8:
        return False
    ratio = max(mean_a, mean_b) / min(mean_a, mean_b)
    return ratio > thresh

def _plot_panel(ax, x_vals, actual, pred_mean, pred_lo, pred_hi,
                xlabel, ylabel, title, n_mc):
    """Plot one panel; use twin-y axis when scales differ greatly."""
    mismatch = _scale_mismatch(actual, pred_mean)

    if mismatch:
        # actual on left axis, prediction on right axis
        ax.set_title(title + "  ⚠ scale mismatch — dual y-axis", fontsize=10, color="darkorange")
        color_a = "royalblue"
        color_p = "crimson"
        l1, = ax.plot(x_vals, actual, color=color_a, lw=2, label="Actual (left)")
        ax.set_ylabel(f"Actual  [{ylabel}]", color=color_a)
        ax.tick_params(axis="y", labelcolor=color_a)

        ax2 = ax.twinx()
        l2, = ax2.plot(x_vals, pred_mean, color=color_p, lw=1.5,
                       linestyle="--", label=f"Pred mean ({n_mc}-MC)  (right)")
        ax2.fill_between(x_vals, pred_lo, pred_hi,
                         color=color_p, alpha=0.18, label="95% CI  (right)")
        ax2.set_ylabel(f"Prediction  [{ylabel}]", color=color_p)
        ax2.tick_params(axis="y", labelcolor=color_p)

        lines = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=8, loc="upper left")
    else:
        ax.set_title(title, fontsize=11)
        ax.plot(x_vals, actual, color="royalblue", lw=2, label="Actual")
        ax.plot(x_vals, pred_mean, color="crimson", lw=1.5,
                linestyle="--", label=f"Pred mean ({n_mc}-MC)")
        ax.fill_between(x_vals, pred_lo, pred_hi,
                        color="crimson", alpha=0.18, label="95% CI")
        ax.legend(fontsize=9)
        ax.set_ylabel(ylabel)

    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.3)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(13, 5))
fig.suptitle(
    f"MTGNN {H}-month Forecast — {node_names[ni]}  |  {feature_names[fi]}",
    fontsize=13, fontweight="bold",
)

date_fmt = mdates.DateFormatter("%Y-%m")

_plot_panel(
    ax,
    x_vals    = dates_full,
    actual    = actual_full,
    pred_mean = pred_full_mean,
    pred_lo   = pred_full_lo,
    pred_hi   = pred_full_hi,
    xlabel    = "Month",
    ylabel    = feature_names[fi],
    title     = (f"Input: {to_dt(all_dates[test_start_idx + last]).strftime('%Y-%m')} ~ "
                 f"{to_dt(all_dates[test_start_idx + last + L - 1]).strftime('%Y-%m')}  →  "
                 f"Forecast: {dates_full[0].strftime('%Y-%m')} ~ {dates_full[-1].strftime('%Y-%m')}  "
                 f"({args.n_mc}-MC dropout)"),
    n_mc      = args.n_mc,
)
ax.xaxis.set_major_formatter(date_fmt)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig(args.out, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {args.out}")
