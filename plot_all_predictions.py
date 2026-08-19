"""
Generate all prediction plots for the US Industrial Landscape dataset.

Outputs (all saved to plots/):
  1. per_node/node{i:02d}_*.png   — 10 figures, one per node (2×5 feature grid)
  2. per_feature/feat{j:02d}_*.png — 10 figures, one per feature (2×5 node grid)
  3. summary_mape_heatmap.png     — N×F MAPE heatmap (paper-ready summary)

Efficiency: MC-Dropout inference is run ONCE for ALL (N, F) combinations,
storing (n_mc, S, H, N, F) predictions in memory, then plots are generated
by slicing — no repeated forward passes.

Usage:
    python plot_all_predictions.py                 # all outputs
    python plot_all_predictions.py --only heatmap  # summary heatmap only
    python plot_all_predictions.py --only node     # per-node grids only
    python plot_all_predictions.py --only feature  # per-feature grids only
    python plot_all_predictions.py --n_mc 10       # faster (less MC samples)
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
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
parser.add_argument("--n_mc",      type=int, default=20,
                    help="MC-Dropout passes (20 is a good speed/quality balance)")
parser.add_argument("--only",      type=str, default="all",
                    choices=["all", "node", "feature", "heatmap"],
                    help="which outputs to generate")
parser.add_argument("--out_dir",   type=str, default="plots")
args = parser.parse_args()

if args.device.startswith("cuda") and not torch.cuda.is_available():
    args.device = "cpu"
device = torch.device(args.device)

# ── Load metadata ─────────────────────────────────────────────────────────────
data_dir   = Path(args.data_dir)
out_dir    = Path(args.out_dir)
meta       = json.loads((data_dir / "meta.json").read_text())
node_names    = meta["node_names"]
feature_names = meta["feature_names"]
N_nodes = len(node_names)
N_feats = len(feature_names)

_scaler = data_dir / "scaler_trainval.npz"
if not _scaler.exists():
    _scaler = data_dir / "scaler.npz"
scaler_npz  = np.load(_scaler)
scaler_mean = torch.tensor(scaler_npz["mean"], dtype=torch.float32).to(device)  # (N, F)
scaler_std  = torch.tensor(scaler_npz["std"],  dtype=torch.float32).to(device)  # (N, F)

test_npz = np.load(data_dir / "test.npz")
x_test   = torch.tensor(test_npz["x"], dtype=torch.float32)   # (S, L, N, F)
y_test   = torch.tensor(test_npz["y"], dtype=torch.float32)   # (S, H, N, F)
S        = len(x_test)

L, H = args.seq_in, args.seq_out
OUT_DIM = H * N_feats

# ── Calendar dates for the last test sample's H-step targets ─────────────────
full_npz  = np.load("US_Industrial_Landscape_complete_tensor.npz")
all_dates = full_npz["dates"]
T_total   = len(all_dates)
n_samples = T_total - L - H + 1
n_train   = int(n_samples * 0.70)
n_val     = int(n_samples * 0.15)
test_start_idx = n_train + n_val

last = S - 1
def to_dt(s):
    return datetime.strptime(str(s)[:7], "%Y-%m")

dates_full = [to_dt(all_dates[test_start_idx + last + L + h]) for h in range(H)]
input_start = to_dt(all_dates[test_start_idx + last])
input_end   = to_dt(all_dates[test_start_idx + last + L - 1])

# ── Load model ────────────────────────────────────────────────────────────────
model = gtnet(
    gcn_true=True, buildA_true=True, gcn_depth=2,
    num_nodes=N_nodes, device=device, predefined_A=None,
    dropout=0.3, subgraph_size=10, node_dim=40, dilation_exponential=1,
    conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128,
    seq_length=L, in_dim=N_feats, out_dim=OUT_DIM, layers=3,
    propalpha=0.05, tanhalpha=3.0, layer_norm_affline=True,
    horizon_decoder="gru_ctx_hemb", n_horizon=H, n_features=N_feats,
    feature_graph=args.feature_graph,
).to(device)
model.load_state_dict(torch.load(args.ckpt, map_location=device))
model.train()   # keep dropout active for MC

# ── MC-Dropout inference — ONE pass for ALL nodes & features ─────────────────
print(f"Running {args.n_mc} MC-Dropout passes for last test sample ...", flush=True)
mc_runs = []   # each entry: (H, N, F) raw predictions
with torch.no_grad():
    for mc in range(args.n_mc):
        xb  = x_test[last:last+1].to(device)              # (1, L, N, F)
        inp = xb.permute(0, 3, 2, 1)                       # (1, F, N, L)
        out = model(inp)                                    # (1, H*F, N, 1)
        out = out.squeeze(-1).view(1, H, N_feats, N_nodes).permute(0, 1, 3, 2)  # (1,H,N,F)
        pred_raw = (out * scaler_std + scaler_mean).squeeze(0).cpu().numpy()    # (H,N,F)
        mc_runs.append(pred_raw)
        if (mc + 1) % 5 == 0:
            print(f"  {mc+1}/{args.n_mc}", flush=True)

mc_arr = np.stack(mc_runs, axis=0)   # (n_mc, H, N, F)
actual = y_test[last].numpy()        # (H, N, F)  raw scale

pred_mean = mc_arr.mean(axis=0)                          # (H, N, F)
pred_lo   = np.percentile(mc_arr,  2.5, axis=0)
pred_hi   = np.percentile(mc_arr, 97.5, axis=0)

print("Inference done.\n", flush=True)

# ── Compute per-(node, feature) MAPE for heatmap ─────────────────────────────
safe_actual = actual.copy()
safe_actual[np.abs(safe_actual) < 1e-8] = np.nan
mape_nf = np.nanmean(
    np.abs(pred_mean - actual) / np.abs(safe_actual) * 100,
    axis=0
)   # (N, F)

# ── Helpers ───────────────────────────────────────────────────────────────────
date_fmt    = mdates.DateFormatter("%Y-%m")
date_locator = mdates.MonthLocator(interval=6)
COLORS = dict(actual="royalblue", pred="crimson", ci="crimson")

def _check_scale_mismatch(a, b, thresh=3.0):
    ma, mb = abs(np.nanmean(a)), abs(np.nanmean(b))
    if ma < 1e-8 or mb < 1e-8:
        return False
    return max(ma, mb) / min(ma, mb) > thresh

def _draw_forecast(ax, act, pm, plo, phi, mape_val, ylabel="", show_xlabel=True):
    """Draw a single forecast panel onto ax."""
    mismatch = _check_scale_mismatch(act, pm)

    if mismatch:
        ax2 = ax.twinx()
        ax.plot(dates_full, act, color=COLORS["actual"], lw=1.5, label="Actual")
        ax2.plot(dates_full, pm,  color=COLORS["pred"],   lw=1.2, ls="--", label="Pred")
        ax2.fill_between(dates_full, plo, phi, color=COLORS["ci"], alpha=0.15)
        ax.set_ylabel(ylabel, fontsize=7, color=COLORS["actual"])
        ax2.set_ylabel("Pred", fontsize=7, color=COLORS["pred"])
        ax.tick_params(axis="y", labelsize=6, labelcolor=COLORS["actual"])
        ax2.tick_params(axis="y", labelsize=6, labelcolor=COLORS["pred"])
        ax.set_title(f"MAPE={mape_val:.1f}%  ⚠dual-y", fontsize=7, color="darkorange")
    else:
        ax.plot(dates_full, act, color=COLORS["actual"], lw=1.5, label="Actual")
        ax.plot(dates_full, pm,  color=COLORS["pred"],   lw=1.2, ls="--", label="Pred")
        ax.fill_between(dates_full, plo, phi, color=COLORS["ci"], alpha=0.15)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(f"MAPE={mape_val:.1f}%", fontsize=8)

    ax.xaxis.set_major_formatter(date_fmt)
    ax.xaxis.set_major_locator(date_locator)
    if show_xlabel:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=6)
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(alpha=0.25)

SHORT_FEAT = [
    "Employment (K)", "Weekly hrs", "Real hrly wage",
    "Job openings %", "Hires %", "Quits %",
    "Layoffs %", "Biz apps /10K", "High-prop share", "Wage app share",
]
SHORT_NODE = [n.split(" and ")[0].split(",")[0][:18] for n in node_names]

# ── 1. Per-node figures (2 rows × 5 cols, all 10 features) ───────────────────
if args.only in ("all", "node"):
    node_dir = out_dir / "per_node"
    node_dir.mkdir(parents=True, exist_ok=True)
    for ni, nn in enumerate(node_names):
        fig, axes = plt.subplots(2, 5, figsize=(18, 7))
        fig.suptitle(
            f"MTGNN 36-month Forecast  —  {nn}\n"
            f"Input: {input_start.strftime('%Y-%m')} ~ {input_end.strftime('%Y-%m')}  "
            f"→  Forecast: {dates_full[0].strftime('%Y-%m')} ~ {dates_full[-1].strftime('%Y-%m')}",
            fontsize=11, fontweight="bold",
        )
        for fi in range(N_feats):
            ax = axes[fi // 5][fi % 5]
            _draw_forecast(
                ax,
                act       = actual[:, ni, fi],
                pm        = pred_mean[:, ni, fi],
                plo       = pred_lo[:, ni, fi],
                phi       = pred_hi[:, ni, fi],
                mape_val  = mape_nf[ni, fi],
                ylabel    = SHORT_FEAT[fi],
                show_xlabel = (fi >= 5),
            )
            ax.set_title(f"{SHORT_FEAT[fi]}\nMAPE={mape_nf[ni, fi]:.1f}%", fontsize=8)
        # shared legend in top-right
        handles = [
            plt.Line2D([0],[0], color=COLORS["actual"], lw=1.5, label="Actual"),
            plt.Line2D([0],[0], color=COLORS["pred"],   lw=1.2, ls="--", label=f"Pred ({args.n_mc}-MC)"),
            plt.Rectangle((0,0),1,1, fc=COLORS["ci"], alpha=0.3, label="95% CI"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
                   bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.03, 1, 1])
        slug = nn.replace(" ", "_").replace(",", "").replace("/", "-")[:30]
        fname = node_dir / f"node{ni:02d}_{slug}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}", flush=True)

# ── 2. Per-feature figures (2 rows × 5 cols, all 10 nodes) ───────────────────
if args.only in ("all", "feature"):
    feat_dir = out_dir / "per_feature"
    feat_dir.mkdir(parents=True, exist_ok=True)
    for fi, fn in enumerate(feature_names):
        fig, axes = plt.subplots(2, 5, figsize=(18, 7))
        fig.suptitle(
            f"MTGNN 36-month Forecast  —  {fn}\n"
            f"Input: {input_start.strftime('%Y-%m')} ~ {input_end.strftime('%Y-%m')}  "
            f"→  Forecast: {dates_full[0].strftime('%Y-%m')} ~ {dates_full[-1].strftime('%Y-%m')}",
            fontsize=11, fontweight="bold",
        )
        for ni in range(N_nodes):
            ax = axes[ni // 5][ni % 5]
            _draw_forecast(
                ax,
                act       = actual[:, ni, fi],
                pm        = pred_mean[:, ni, fi],
                plo       = pred_lo[:, ni, fi],
                phi       = pred_hi[:, ni, fi],
                mape_val  = mape_nf[ni, fi],
                ylabel    = SHORT_NODE[ni],
                show_xlabel = (ni >= 5),
            )
            ax.set_title(f"{SHORT_NODE[ni]}\nMAPE={mape_nf[ni, fi]:.1f}%", fontsize=8)
        handles = [
            plt.Line2D([0],[0], color=COLORS["actual"], lw=1.5, label="Actual"),
            plt.Line2D([0],[0], color=COLORS["pred"],   lw=1.2, ls="--", label=f"Pred ({args.n_mc}-MC)"),
            plt.Rectangle((0,0),1,1, fc=COLORS["ci"], alpha=0.3, label="95% CI"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
                   bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.03, 1, 1])
        slug = fn.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")[:30]
        fname = feat_dir / f"feat{fi:02d}_{slug}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}", flush=True)

# ── 3. MAPE heatmap (paper-ready summary) ────────────────────────────────────
if args.only in ("all", "heatmap"):
    fig, ax = plt.subplots(figsize=(13, 5))

    im = ax.imshow(mape_nf, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=50)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("MAPE (%)", fontsize=10)

    ax.set_xticks(range(N_feats))
    ax.set_xticklabels(SHORT_FEAT, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(N_nodes))
    ax.set_yticklabels(SHORT_NODE, fontsize=9)

    # annotate each cell
    for ni in range(N_nodes):
        for fi in range(N_feats):
            val = mape_nf[ni, fi]
            color = "white" if val > 35 else "black"
            ax.text(fi, ni, f"{val:.1f}", ha="center", va="center",
                    fontsize=7.5, color=color, fontweight="bold")

    mean_row = mape_nf.mean(axis=1)   # per-node mean MAPE
    mean_col = mape_nf.mean(axis=0)   # per-feature mean MAPE
    overall  = mape_nf.mean()

    ax.set_title(
        f"MTGNN 36-month Forecast — MAPE (%) per Node × Feature\n"
        f"Input: {input_start.strftime('%Y-%m')} ~ {input_end.strftime('%Y-%m')}  "
        f"→  Forecast: {dates_full[0].strftime('%Y-%m')} ~ {dates_full[-1].strftime('%Y-%m')}  "
        f"|  Overall mean MAPE = {overall:.1f}%",
        fontsize=10, fontweight="bold",
    )

    # add row/col mean annotations as extra text on the right/bottom
    for ni, mv in enumerate(mean_row):
        ax.text(N_feats + 0.15, ni, f"avg={mv:.1f}%", va="center", fontsize=7.5, color="gray")
    ax.set_xlim(-0.5, N_feats + 1.2)

    fig.tight_layout()
    fname = out_dir / "summary_mape_heatmap.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}", flush=True)

print("\nAll done.")
