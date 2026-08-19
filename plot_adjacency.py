"""
Visualise the learned adaptive adjacency matrix from the trained MTGNN model.

Outputs (saved to plots/adjacency/):
  adjacency_sparse.png   — the actual top-k A used during GCN (sparse)
  adjacency_full.png     — the dense A before top-k sparsification
  adjacency_asymmetry.png — A - Aᵀ: who leads whom (net directional influence)
  adjacency_influence.png — bar charts: out-degree vs in-degree per sector
  adjacency_vs_corr.png  — learned A vs Pearson correlation of training data
  adjacency_combined.png — paper-ready combined figure

Formula (from layer.py graph_constructor.forward):
  nv1 = tanh(α · lin1(emb1(idx)))
  nv2 = tanh(α · lin2(emb2(idx)))
  a   = nv1 @ nv2ᵀ  -  nv2 @ nv1ᵀ    ← antisymmetric by construction
  A   = relu(tanh(α · a))              ← A[i,j]>0 ⟹ A[j,i]=0 (guaranteed)

Usage:
    python plot_adjacency.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

from net import gtnet

# ── Config ────────────────────────────────────────────────────────────────────
CKPT     = "checkpoints/industrial/mtgnn_fgraph_s1_L12_H24.pth"
DATA_DIR = Path("data/industrial_L12_H24")
OUT_DIR  = Path("plots/adjacency")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N, F_dim, L, H = 10, 10, 12, 24

meta          = json.loads((DATA_DIR / "meta.json").read_text())
node_names    = meta["node_names"]
SHORT_NODE    = [n.split(" and ")[0].split(",")[0][:18] for n in node_names]

# ── Load model ────────────────────────────────────────────────────────────────
model = gtnet(
    gcn_true=True, buildA_true=True, gcn_depth=2,
    num_nodes=N, device=DEVICE, predefined_A=None,
    dropout=0.3, subgraph_size=10, node_dim=40, dilation_exponential=1,
    conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128,
    seq_length=L, in_dim=F_dim, out_dim=H * F_dim, layers=3,
    propalpha=0.05, tanhalpha=3.0, layer_norm_affline=True,
    horizon_decoder="gru_ctx_hemb", n_horizon=H, n_features=F_dim,
    feature_graph=True,
).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

# ── Extract adjacency matrices ────────────────────────────────────────────────
idx = model.idx   # (N,) node indices
with torch.no_grad():
    A_sparse = model.gc(idx).cpu().numpy()         # top-k sparsified   (N, N)
    A_full   = model.gc.fullA(idx).cpu().numpy()   # dense (no top-k)   (N, N)

A_asym = A_sparse - A_sparse.T   # net asymmetry: positive = net sender, negative = net receiver

print(f"A_sparse  min={A_sparse.min():.4f}  max={A_sparse.max():.4f}  nnz={np.count_nonzero(A_sparse)}/{N*N}")
print(f"A_full    min={A_full.min():.4f}  max={A_full.max():.4f}")
print()

# ── Compute Pearson correlation of training X (per-node mean series) ─────────
train_npz = np.load(DATA_DIR / "train.npz")
scaler    = np.load(DATA_DIR / "scaler.npz")
x_train   = train_npz["x"]   # (S, L, N, F) normalised

# aggregate to one series per node: mean over features then mean over time
# shape: (S*L, N)  — each row is a snapshot, each col is a node
x_flat = x_train.reshape(-1, N, F_dim).mean(axis=2)   # (S*L, N)
corr   = np.corrcoef(x_flat.T)                         # (N, N)

# ── Helper: draw heatmap ──────────────────────────────────────────────────────
def draw_heatmap(ax, mat, title, cmap="RdBu_r", vmin=None, vmax=None,
                 annotate=True, xlabel="Source (j)", ylabel="Target (i)"):
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(SHORT_NODE, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(SHORT_NODE, fontsize=7)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    if annotate:
        for i in range(N):
            for j in range(N):
                v = mat[i, j]
                if abs(v) > 0.01:
                    color = "white" if abs(v) > 0.6 * (vmax or mat.max()) else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=5.5, color=color)

# ── 1. Sparse A (actual GCN graph) ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
draw_heatmap(ax, A_sparse, "Learned Adaptive Adjacency A  (top-k sparse)\nA[i,j]: strength of j → i influence",
             cmap="YlOrRd", vmin=0, vmax=A_sparse.max())
ax.set_xlabel("Source sector  j  (information flows FROM here)", fontsize=8)
ax.set_ylabel("Target sector  i  (prediction of THIS node)", fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "adjacency_sparse.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: adjacency_sparse.png")

# ── 2. Full A (dense, before top-k) ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
draw_heatmap(ax, A_full, "Full (dense) Adaptive Adjacency  (before top-k sparsification)",
             cmap="YlOrRd", vmin=0, vmax=A_full.max())
fig.tight_layout()
fig.savefig(OUT_DIR / "adjacency_full.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: adjacency_full.png")

# ── 3. Asymmetry: A - Aᵀ ─────────────────────────────────────────────────────
vabs = np.abs(A_asym).max()
fig, ax = plt.subplots(figsize=(8, 7))
draw_heatmap(ax, A_asym,
             "Net Directional Asymmetry  A − Aᵀ\n"
             "Red (i,j)>0: j → i  stronger than  i → j\n"
             "Blue (i,j)<0: i → j  stronger than  j → i",
             cmap="RdBu_r", vmin=-vabs, vmax=vabs)
fig.tight_layout()
fig.savefig(OUT_DIR / "adjacency_asymmetry.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: adjacency_asymmetry.png")

# ── 4. Influence bars (out-degree vs in-degree) ───────────────────────────────
out_strength = A_sparse.sum(axis=1)   # row sum:  how much i sends to others
in_strength  = A_sparse.sum(axis=0)   # col sum:  how much others send to i
net_strength = out_strength - in_strength

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
x_pos = np.arange(N)
axes[0].barh(x_pos, out_strength, color="tomato")
axes[0].set_yticks(x_pos); axes[0].set_yticklabels(SHORT_NODE, fontsize=8)
axes[0].set_xlabel("Sum of outgoing weights"); axes[0].set_title("Out-strength  (sender role)", fontweight="bold")
axes[0].invert_yaxis()

axes[1].barh(x_pos, in_strength, color="steelblue")
axes[1].set_yticks(x_pos); axes[1].set_yticklabels(SHORT_NODE, fontsize=8)
axes[1].set_xlabel("Sum of incoming weights"); axes[1].set_title("In-strength  (receiver role)", fontweight="bold")
axes[1].invert_yaxis()

colors = ["tomato" if v >= 0 else "steelblue" for v in net_strength]
axes[2].barh(x_pos, net_strength, color=colors)
axes[2].set_yticks(x_pos); axes[2].set_yticklabels(SHORT_NODE, fontsize=8)
axes[2].axvline(0, color="black", lw=0.8)
axes[2].set_xlabel("Net influence (out − in)"); axes[2].set_title("Net role  (red=sender / blue=receiver)", fontweight="bold")
axes[2].invert_yaxis()

fig.suptitle("Sector Influence in Learned Graph  (MTGNN adaptive adjacency)", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "adjacency_influence.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: adjacency_influence.png")

# ── 5. Learned A vs Pearson correlation ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
draw_heatmap(axes[0], A_sparse,
             "Learned A  (directional, predictive)\nA[i,j]: j helps predict i",
             cmap="YlOrRd", vmin=0, vmax=A_sparse.max())
draw_heatmap(axes[1], corr,
             "Pearson Correlation  (undirected, contemporaneous)\ncorr[i,j]: co-movement strength",
             cmap="RdBu_r", vmin=-1, vmax=1)
fig.suptitle("Learned Adjacency vs. Pearson Correlation\n"
             "Divergences reveal predictive structure beyond simple co-movement",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "adjacency_vs_corr.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: adjacency_vs_corr.png")

# ── 6. Paper-ready combined figure ───────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1:])

draw_heatmap(ax1, A_sparse,
             "(a) Sparse A  (top-k GCN graph)\nA[i,j]: j → i influence",
             cmap="YlOrRd", vmin=0, vmax=A_sparse.max())

draw_heatmap(ax2, A_full,
             "(b) Dense A  (before sparsification)",
             cmap="YlOrRd", vmin=0, vmax=A_full.max())

draw_heatmap(ax3, A_asym,
             "(c) Net asymmetry  A − Aᵀ\nRed=receiver, Blue=sender",
             cmap="RdBu_r", vmin=-vabs, vmax=vabs)

draw_heatmap(ax4, corr,
             "(d) Pearson Correlation  (baseline)",
             cmap="RdBu_r", vmin=-1, vmax=1)

# influence bar in ax5 (horizontal, combined out/in)
w = 0.35
x_pos2 = np.arange(N)
ax5.bar(x_pos2 - w/2, out_strength, width=w, label="Out-strength (sender)", color="tomato", alpha=0.85)
ax5.bar(x_pos2 + w/2, in_strength,  width=w, label="In-strength (receiver)", color="steelblue", alpha=0.85)
ax5.set_xticks(x_pos2)
ax5.set_xticklabels(SHORT_NODE, rotation=40, ha="right", fontsize=8)
ax5.set_ylabel("Summed adjacency weight")
ax5.set_title("(e) Out-strength vs In-strength per Sector", fontweight="bold", fontsize=9)
ax5.legend(fontsize=8)
ax5.grid(axis="y", alpha=0.3)

fig.suptitle(
    "MTGNN Learned Adaptive Adjacency Analysis\n"
    f"Training period: 2010 ~ 2022  |  10 industry sectors  |  tanhalpha=3.0  subgraph_size=10",
    fontsize=12, fontweight="bold",
)
fig.savefig(OUT_DIR / "adjacency_combined.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: adjacency_combined.png")

# ── Print textual summary ─────────────────────────────────────────────────────
print("\n=== Top 5 strongest directed edges (A_sparse) ===")
flat = [(A_sparse[i,j], i, j) for i in range(N) for j in range(N) if i != j]
flat.sort(reverse=True)
for val, i, j in flat[:10]:
    if val > 0:
        print(f"  {SHORT_NODE[j]:20s} → {SHORT_NODE[i]:20s}  weight={val:.4f}")

print("\n=== Net sender (positive) / receiver (negative) roles ===")
order = np.argsort(net_strength)[::-1]
for idx_ in order:
    role = "SENDER  " if net_strength[idx_] >= 0 else "receiver"
    print(f"  {SHORT_NODE[idx_]:20s}  net={net_strength[idx_]:+.4f}  [{role}]")
