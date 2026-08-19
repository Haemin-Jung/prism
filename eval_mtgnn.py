"""
eval_mtgnn.py — Standalone evaluation of a trained MTGNN checkpoint.

Loads the best checkpoint, runs inference on the test set, and prints
the full five-metric comparison table (MAE, RMSE, MAPE, RSE, MASE).
Results can also be saved as JSON for later merging with baseline results.

Usage examples
──────────────
  # deterministic only
  python eval_mtgnn.py

  # MC-Dropout (B-MTGNN) only
  python eval_mtgnn.py --n_mc 30

  # both MTGNN and B-MTGNN in one table  ← recommended
  python eval_mtgnn.py --n_mc 30 --both

  # merge with baseline results and show full comparison table
  python eval_mtgnn.py --n_mc 30 --both --merge results/baselines.json --save results/all.json --save_excel results/all.xlsx
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from net import gtnet
from metrics import evaluate, naive_scale, print_results_table, \
                   save_results_json, load_results_json, save_results_excel
from data_utils import fit_targets, load_meta, load_scaler, load_split

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",     default="checkpoints/industrial/mtgnn_fgraph_s1_L12_H24.pth",
                    help="path to saved model checkpoint")
parser.add_argument("--data_dir", default="data/industrial_L12_H24")
parser.add_argument("--fit_on",   default="trainval", choices=("train", "trainval"),
                    help="normalisation / MASE regime the checkpoint was trained "
                         "under. Use trainval for checkpoints produced with "
                         "train_industrial.py --refit_trainval.")
parser.add_argument("--device",   default="cuda:0")
parser.add_argument("--n_mc",     type=int, default=0,
                    help="MC-Dropout passes (0 = deterministic). "
                         "Use ≥20 for stable uncertainty estimates.")
parser.add_argument("--batch",    type=int, default=32)
parser.add_argument("--horizons", default="3,6,12,24")
parser.add_argument("--detail_h", type=int, default=0,
                    help="horizon for the detailed five-metric columns (0 = auto: best for target model)")
parser.add_argument("--save",     default="",
                    help="write results JSON to this path (optional)")
parser.add_argument("--save_excel", default="",
                    help="also write results to an Excel file (.xlsx)")
parser.add_argument("--merge",    default="",
                    help="path to baseline results JSON to merge into table")
parser.add_argument("--both",     action="store_true",
                    help="run both deterministic MTGNN and B-MTGNN (MC-Dropout) "
                         "and show both in the same table. requires --n_mc > 0")
# model architecture (must match the checkpoint being evaluated)
parser.add_argument("--conv_ch",   type=int, default=32)
parser.add_argument("--residual_ch",type=int, default=-1,
                    help="residual channels (default: same as conv_ch)")
parser.add_argument("--skip_ch",   type=int, default=64)
parser.add_argument("--end_ch",    type=int, default=128)
parser.add_argument("--node_dim",  type=int, default=40)
parser.add_argument("--gcn_depth", type=int, default=2)
parser.add_argument("--layers",    type=int, default=3)
parser.add_argument("--dilation_exp", type=int, default=1,
                    help="dilation exponential factor (must match training)")
parser.add_argument("--dropout",   type=float, default=0.3)
parser.add_argument("--label_suffix", default="",
                    help="appended to model names in the results table "
                         "(e.g. ' RF43')")
parser.add_argument("--horizon_decoder", default="gru_ctx_hemb",
                    choices=("gru_ctx_hemb",),
                    help="must match training")
parser.add_argument("--gru_hidden", type=int, default=0)
parser.add_argument("--horizon_emb_dim", type=int, default=8)
parser.add_argument("--feature_graph", action="store_true",
                    help="must match training")
parser.add_argument("--feat_gcn_depth", type=int, default=0)
parser.add_argument("--subgraph_size", type=int, default=-1,
                    help="top-k neighbours per node (default: num_nodes)")
parser.add_argument("--propalpha",  type=float, default=0.05)
parser.add_argument("--tanhalpha",  type=float, default=3.0)
args = parser.parse_args()

if args.device.startswith("cuda") and not torch.cuda.is_available():
    args.device = "cpu"
device   = torch.device(args.device)
horizons = [int(h) for h in args.horizons.split(",")]

# ── Data ──────────────────────────────────────────────────────────────────────
data_dir = Path(args.data_dir)
if not data_dir.exists():
    raise FileNotFoundError(
        f"Data directory not found: {data_dir}\n"
        f"Run: python prepare_industrial.py --seq_in 12 --seq_out 24"
    )

meta      = load_meta(data_dir)
L, H      = meta["seq_in"], meta["seq_out"]
N, F_dim  = meta["n_nodes"], meta["n_features"]
N_model = N
F_model = F_dim
if args.subgraph_size == -1:
    subgraph_size = N_model
else:
    subgraph_size = min(args.subgraph_size, N_model)

smean, sstd     = load_scaler(data_dir, args.fit_on)   # (N, F) each
_x_test, y_test = load_split(data_dir, "test", args.fit_on)
x_test    = torch.tensor(_x_test, dtype=torch.float32)
ns        = naive_scale(fit_targets(data_dir, args.fit_on))  # (N, F)

print(f"Device      : {device}")
print(f"Checkpoint  : {args.ckpt}")
print(f"Fit on      : {args.fit_on}")
print(f"Test samples: {len(x_test)}  (L={L}, H={H}, N={N}, F={F_dim})")
if args.both and args.n_mc > 0:
    print(f"Eval mode   : deterministic  +  MC-Dropout ×{args.n_mc}")
elif args.n_mc > 0:
    print(f"Eval mode   : MC-Dropout ×{args.n_mc}")
else:
    print(f"Eval mode   : deterministic")

# ── Model ─────────────────────────────────────────────────────────────────────
ckpt_path = Path(args.ckpt)
if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

model = gtnet(
    gcn_true=True, buildA_true=True, gcn_depth=args.gcn_depth,
    num_nodes=N_model, device=device, predefined_A=None,
    dropout=args.dropout,
    subgraph_size=subgraph_size,
    node_dim=args.node_dim, dilation_exponential=args.dilation_exp,
    conv_channels=args.conv_ch,
    residual_channels=args.conv_ch if args.residual_ch == -1 else args.residual_ch,
    skip_channels=args.skip_ch, end_channels=args.end_ch,
    seq_length=L, in_dim=F_model, out_dim=H * F_model, layers=args.layers,
    propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True,
    horizon_decoder=args.horizon_decoder, n_horizon=H, n_features=F_model,
    gru_hidden=args.gru_hidden, horizon_emb_dim=args.horizon_emb_dim,
    feature_graph=args.feature_graph, feat_gcn_depth=args.feat_gcn_depth,
).to(device)
print(f"Architecture : layers={args.layers}  dil_exp={args.dilation_exp}  "
      f"RF={model.receptive_field}  decoder={args.horizon_decoder}"
      f"  feature_graph={args.feature_graph}"
      f"  subgraph_size={subgraph_size}")
model.load_state_dict(torch.load(ckpt_path, map_location=device))

smean_t = torch.tensor(smean, device=device)
sstd_t  = torch.tensor(sstd,  device=device)

# ── Inference helper ──────────────────────────────────────────────────────────
def _forward_raw(xb: torch.Tensor) -> np.ndarray:
    """Single forward pass → (B, H, N, F) raw scale."""
    B = xb.size(0)
    inp = xb.permute(0, 3, 2, 1).to(device)
    out = model(inp)
    pred_norm = out.squeeze(-1).view(B, H, F_dim, N).permute(0, 1, 3, 2)
    return (pred_norm * sstd_t + smean_t).cpu().numpy()

def _run_inference(n_mc: int) -> np.ndarray:
    """Run full test-set inference. n_mc=0 → deterministic, >0 → MC mean."""
    if n_mc > 0:
        model.train()
    else:
        model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(x_test), args.batch):
            xb = x_test[i : i + args.batch]
            if n_mc > 0:
                runs = np.stack([_forward_raw(xb) for _ in range(n_mc)], axis=0)
                parts.append(runs.mean(axis=0))
            else:
                parts.append(_forward_raw(xb))
    return np.concatenate(parts, axis=0)  # (S, H, N, F)

# ── Run inference (one or both modes) ─────────────────────────────────────────
run_det = (args.n_mc == 0) or args.both
run_mc  = args.n_mc > 0

entries = {}   # label → preds_raw

sfx = args.label_suffix

if run_det:
    print("  Running deterministic inference …")
    entries[f"MTGNN{sfx}"] = _run_inference(0)

if run_mc:
    print(f"  Running MC-Dropout ×{args.n_mc} inference …")
    entries[f"B-MTGNN{sfx} (MC={args.n_mc})"] = _run_inference(args.n_mc)

# ── Metrics ───────────────────────────────────────────────────────────────────
new_results = {
    label: evaluate(preds, y_test, smean, sstd, ns, horizons)
    for label, preds in entries.items()
}

# The "featured" model for auto-selecting detail_h
# prefer B-MTGNN when both are present
target_label = (f"B-MTGNN{sfx} (MC={args.n_mc})" if run_mc else f"MTGNN{sfx}")

# ── Build combined table ───────────────────────────────────────────────────────
all_results = {}
if args.merge and Path(args.merge).exists():
    all_results.update(load_results_json(args.merge, fit_on=args.fit_on))
all_results.update(new_results)

print_results_table(
    all_results,
    detail_h=args.detail_h if args.detail_h > 0 else None,
    horizons=horizons,
    title="Forecasting Performance Comparison",
    target_model=target_label,
)

if args.save:
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    save_results_json(all_results, args.save, fit_on=args.fit_on)

if args.save_excel:
    Path(args.save_excel).parent.mkdir(parents=True, exist_ok=True)
    save_results_excel(
        all_results, args.save_excel, horizons=horizons,
        detail_h=args.detail_h if args.detail_h > 0 else None,
        target_model=target_label,
    )
