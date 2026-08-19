"""
Train MTGNN (HGRU-CTX+HE, optional feature graph) on US Industrial Landscape.

Default setting: L=12 lookback, H=24 horizon, lr=2e-3, MAE, refit on train+val.

    python train_industrial.py --device cuda:0 --refit_trainval --feature_graph
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from net import gtnet
from metrics import evaluate, naive_scale, print_results_table
from data_utils import fit_targets, load_fit_split, load_meta, load_scaler, load_split

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device",        type=str,   default="cuda:0")
parser.add_argument("--data_dir",      type=str,   default=None,
                    help="directory produced by prepare_industrial.py "
                         "(default: data/industrial_L{seq_in}_H{seq_out})")

# seq lengths (must match what prepare_industrial.py used)
parser.add_argument("--seq_in",        type=int,   default=12,  help="lookback L")
parser.add_argument("--seq_out",       type=int,   default=24,  help="max forecast horizon H")
parser.add_argument("--eval_horizons", type=str,   default="3,6,12,24",
                    help="comma-separated horizons to report at test time")

# graph / model architecture
parser.add_argument("--gcn_depth",     type=int,   default=2)
parser.add_argument("--subgraph_size", type=int,   default=10,  help="k for top-k adjacency; must be <= num_nodes")
parser.add_argument("--node_dim",      type=int,   default=40)
parser.add_argument("--dilation_exp",  type=int,   default=1)
parser.add_argument("--conv_ch",       type=int,   default=32)
parser.add_argument("--residual_ch",   type=int,   default=32)
parser.add_argument("--skip_ch",       type=int,   default=64)
parser.add_argument("--end_ch",        type=int,   default=128)
parser.add_argument("--layers",        type=int,   default=3)
parser.add_argument("--dropout",       type=float, default=0.3)
parser.add_argument("--propalpha",     type=float, default=0.05)
parser.add_argument("--tanhalpha",     type=float, default=3.0)
parser.add_argument("--horizon_decoder", default="gru_ctx_hemb",
                    choices=("gru_ctx_hemb",),
                    help="HGRU-CTX+HE: dh=GRUCell([zi; e_h], d_{h-1})")
parser.add_argument("--gru_hidden",    type=int,   default=0,
                    help="hidden size for the horizon GRU (0 = end_ch)")
parser.add_argument("--horizon_emb_dim", type=int, default=8,
                    help="horizon embedding dim for HGRU-CTX+HE")
parser.add_argument("--feature_graph", action="store_true",
                    help="factorized industry/feature message-passing stem "
                         "(A_I shared with mixprop; learned A_F on features)")
parser.add_argument("--feat_gcn_depth", type=int, default=0,
                    help="hops in feature-graph stem (0 = same as --gcn_depth)")

# training
parser.add_argument("--epochs",        type=int,   default=300)
parser.add_argument("--batch_size",    type=int,   default=16)
parser.add_argument("--lr",            type=float, default=2e-3)
parser.add_argument("--weight_decay",  type=float, default=1e-4)
parser.add_argument("--clip",          type=float, default=5.0)
parser.add_argument("--patience",      type=int,   default=60,
                    help="early-stopping patience in epochs; must leave room for "
                         "several --lr_patience cycles or runs die before a reduced "
                         "learning rate can take effect")
parser.add_argument("--lr_patience",   type=int,   default=10,
                    help="ReduceLROnPlateau patience in epochs")
parser.add_argument("--min_epochs",    type=int,   default=30,
                    help="never early-stop before this epoch; the validation curve "
                         "has a shallow local minimum around epoch 7-8")
parser.add_argument("--seed",          type=int,   default=42)
# loss & augmentation
parser.add_argument("--loss",          type=str,   default="mae",
                    choices=["mae", "mse", "huber"],
                    help="training loss: mae | mse (=RMSE-aligned) | huber")
parser.add_argument("--huber_delta",   type=float, default=1.0,
                    help="delta for Huber loss")
parser.add_argument("--aug_noise",     type=float, default=0.0,
                    help="std of Gaussian noise added to input during training "
                         "(in normalised units; 0 = disabled). Try 0.02–0.05.")
# evaluation protocol
parser.add_argument("--refit_trainval", action="store_true",
                    help="after selecting the epoch count on the validation split, "
                         "refit a fresh model on train+val and evaluate that on test. "
                         "The scaler and the MASE denominator follow train+val too.")
parser.add_argument("--refit_epochs",  type=int,   default=0,
                    help="override the refit epoch count (0 = use the selected one)")
parser.add_argument("--save",          type=str,   default="checkpoints/industrial",
                    help="directory for saved checkpoints")
parser.add_argument("--expid",         type=str,   default=None,
                    help="checkpoint stem (default: mtgnn_s{seed}_L{L}_H{H})")
parser.add_argument("--save_metrics",  type=str,   default="",
                    help="optional JSON path for best-val / test metrics")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ── Device ────────────────────────────────────────────────────────────────────
if args.device.startswith("cuda") and not torch.cuda.is_available():
    print("CUDA not available, falling back to CPU.")
    args.device = "cpu"
device = torch.device(args.device)
print(f"Device : {device}")

# ── Constants ─────────────────────────────────────────────────────────────────
L  = args.seq_in
H  = args.seq_out

# Derive defaults from L / H when not explicitly provided
if args.data_dir is None:
    args.data_dir = f"data/industrial_L{L}_H{H}"
if args.expid is None:
    args.expid = f"mtgnn_s{args.seed}_L{L}_H{H}"

eval_horizons = [int(h) for h in args.eval_horizons.split(",") if int(h) <= H]

# ── Data ──────────────────────────────────────────────────────────────────────
data_dir = Path(args.data_dir)
if not data_dir.exists():
    raise FileNotFoundError(
        f"{data_dir} not found. "
        f"Run: python prepare_industrial.py --seq_in {L} --seq_out {H}"
    )

meta = load_meta(data_dir)
feature_names = meta["feature_names"]
node_names    = meta["node_names"]
N = int(meta["n_nodes"])
F = int(meta["n_features"])
OUT_DIM = H * F
if args.subgraph_size > N:
    args.subgraph_size = N

# The regime the final model is fitted under.  Everything reported at test time
# — the scaler used to invert predictions and the MASE denominator — follows it.
FIT_ON = "trainval" if args.refit_trainval else "train"


def _tensors(pair):
    return (torch.tensor(pair[0], dtype=torch.float32),
            torch.tensor(pair[1], dtype=torch.float32))


def scaler_tensors(fit_on: str):
    mean, std = load_scaler(data_dir, fit_on)
    return (torch.tensor(mean, dtype=torch.float32).to(device),
            torch.tensor(std,  dtype=torch.float32).to(device))


# Phase 1 selects hyper-parameters against the validation split, so it must not
# see validation statistics: both the data and the scaler come from `train`.
x_train, y_train = _tensors(load_split(data_dir, "train", "train"))
x_val,   y_val   = _tensors(load_split(data_dir, "val",   "train"))
sel_mean, sel_std = scaler_tensors("train")

# Phase 2 (and the test evaluation) uses whichever regime FIT_ON names.
x_fit, y_fit     = _tensors(load_fit_split(data_dir, FIT_ON))
x_test, y_test   = _tensors(load_split(data_dir, "test", FIT_ON))
fit_mean, fit_std = scaler_tensors(FIT_ON)

# seasonal-naïve scale (N, F) — used for MASE at test time
ns = naive_scale(fit_targets(data_dir, FIT_ON))

print(f"Train  : x={tuple(x_train.shape)}  y={tuple(y_train.shape)}")
print(f"Val    : x={tuple(x_val.shape)}    y={tuple(y_val.shape)}")
print(f"Test   : x={tuple(x_test.shape)}   y={tuple(y_test.shape)}")
print(f"Fit on : {FIT_ON}  ->  x={tuple(x_fit.shape)}")


def make_batches(x, y, batch_size: int, shuffle: bool = False):
    n = len(x)
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        b = idx[i : i + batch_size]
        yield x[b], y[b]


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    return gtnet(
        gcn_true        = True,
        buildA_true     = True,     # learn adjacency end-to-end; no prior graph needed
        gcn_depth       = args.gcn_depth,
        num_nodes       = N,
        device          = device,
        predefined_A    = None,
        dropout         = args.dropout,
        subgraph_size   = args.subgraph_size,
        node_dim        = args.node_dim,
        dilation_exponential = args.dilation_exp,
        conv_channels   = args.conv_ch,
        residual_channels= args.residual_ch,
        skip_channels   = args.skip_ch,
        end_channels    = args.end_ch,
        seq_length      = L,
        in_dim          = F,
        out_dim         = OUT_DIM,
        layers          = args.layers,
        propalpha       = args.propalpha,
        tanhalpha       = args.tanhalpha,
        layer_norm_affline = True,
        horizon_decoder = args.horizon_decoder,
        n_horizon       = H,
        n_features      = F,
        gru_hidden      = args.gru_hidden,
        horizon_emb_dim = args.horizon_emb_dim,
        feature_graph   = args.feature_graph,
        feat_gcn_depth  = args.feat_gcn_depth,
    ).to(device)


model = build_model()

n_params = sum(p.numel() for p in model.parameters())
print(f"\nParameters     : {n_params:,}")
print(f"Receptive field: {model.receptive_field}  (seq_in={L})")
print(f"Node layout    : industry nodes  N={N}  F={F}")
print(f"OUT_DIM        : {OUT_DIM}  (H={H} x F={F})")
print(f"subgraph_size  : {args.subgraph_size}")
print(f"Horizon decoder: {args.horizon_decoder}  "
      f"(hidden={args.gru_hidden or args.end_ch}  "
      f"horizon_emb_dim={args.horizon_emb_dim})")
print(f"Feature graph  : {args.feature_graph}"
      + (f"  (feat_gcn_depth={args.feat_gcn_depth or args.gcn_depth})"
         if args.feature_graph else ""))

# ── Loss function ──────────────────────────────────────────────────────────────
if args.loss == "mse":
    _loss_fn = nn.MSELoss()
elif args.loss == "huber":
    _loss_fn = nn.HuberLoss(delta=args.huber_delta)
else:
    _loss_fn = nn.L1Loss()  # MAE

def batch_loss(pred, real):
    return _loss_fn(pred, real)

# ── Forward helpers ───────────────────────────────────────────────────────────
def _model_out(net, xb):
    """Run model and return output in normalised space, shape (B, H, N, F)."""
    inp = xb.permute(0, 3, 2, 1).to(device)          # (B, F, N, L)
    out = net(inp)                                     # (B, H*F, N, 1)
    return out.squeeze(-1).view(xb.size(0), H, F, N).permute(0, 1, 3, 2)


def forward_train(net, xb, yb, s_mean, s_std):
    """Return (pred_norm, real_norm) — both in normalised space for loss."""
    pred_norm = _model_out(net, xb)                                  # (B, H, N, F) normalised
    real_norm = (yb.to(device) - s_mean) / s_std                     # normalise target
    return pred_norm, real_norm


def forward_eval(net, xb, yb, s_mean, s_std):
    """Return (pred_raw, real_raw) — both in original units for interpretable metrics."""
    pred_raw = _model_out(net, xb) * s_std + s_mean                  # inverse-normalise
    real_raw = yb.to(device)                                         # already raw scale
    return pred_raw, real_raw


def train_one_epoch(net, opt, x, y, s_mean, s_std):
    net.train()
    losses = []
    for xb, yb in make_batches(x, y, args.batch_size, shuffle=True):
        opt.zero_grad()
        # ── optional input noise augmentation (normalised space) ───────────────
        if args.aug_noise > 0:
            xb = xb + torch.randn_like(xb) * args.aug_noise
        pred, real = forward_train(net, xb, yb, s_mean, s_std)
        loss = batch_loss(pred, real)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def eval_loss(net, x, y, s_mean, s_std):
    net.eval()
    losses = []
    for xb, yb in make_batches(x, y, args.batch_size):
        pred, real = forward_train(net, xb, yb, s_mean, s_std)
        losses.append(batch_loss(pred, real).item())
    return float(np.mean(losses))




# ── Phase 1: select the epoch count against the validation split ──────────────
save_dir = Path(args.save)
save_dir.mkdir(parents=True, exist_ok=True)
ckpt_path = save_dir / f"{args.expid}.pth"
sel_ckpt_path = save_dir / f"{args.expid}_selection.pth"

print(f"\nCheckpoint : {ckpt_path}")
print(f"Eval horizons: {eval_horizons}")
print(f"Loss fn    : {args.loss}" + (f"  (delta={args.huber_delta})" if args.loss == "huber" else ""))
print(f"Aug noise  : {args.aug_noise}" + ("  (disabled)" if args.aug_noise == 0 else ""))
print(f"Protocol   : " + ("select on val, then refit on train+val"
                          if args.refit_trainval else "train only, early-stopped model"))
print()
print(f"{'Epoch':>6}  {'TrainMAE':>11}  {'ValMAE':>11}  {'Best':>11}  {'LR':>8}  {'Time':>6}")
print(f"         (normalised)  (normalised)  (normalised)")
print("-" * 65)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=args.lr_patience, factor=0.5, min_lr=1e-5
)
if args.patience <= 2 * args.lr_patience:
    print(f"WARNING: --patience {args.patience} leaves room for at most one LR "
          f"reduction at --lr_patience {args.lr_patience}; runs may stop while "
          f"still stuck at the initial learning rate.")

best_val_mae = float("inf")
best_epoch   = 0
patience_cnt = 0
# LR actually used in each epoch, so phase 2 can replay the ReduceLROnPlateau
# trajectory without a validation split to drive it.
lr_history = []

for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    current_lr = optimizer.param_groups[0]["lr"]
    lr_history.append(current_lr)

    tr_mae = train_one_epoch(model, optimizer, x_train, y_train, sel_mean, sel_std)
    vl_mae = eval_loss(model, x_val, y_val, sel_mean, sel_std)
    scheduler.step(vl_mae)

    if vl_mae < best_val_mae:
        best_val_mae = vl_mae
        best_epoch   = epoch
        patience_cnt = 0
        torch.save(model.state_dict(), sel_ckpt_path)
    else:
        patience_cnt += 1

    elapsed = time.time() - t0
    print(f"{epoch:6d}  {tr_mae:11.4f}  {vl_mae:11.4f}  {best_val_mae:11.4f}"
          f"  {current_lr:8.2e}  {elapsed:5.1f}s"
          + (" *" if patience_cnt == 0 else ""), flush=True)

    if patience_cnt >= args.patience and epoch >= args.min_epochs:
        print(f"\nEarly stopping at epoch {epoch} (patience={args.patience}).")
        break

print(f"Selected epoch count : {best_epoch}  (val MAE {best_val_mae:.4f})")

# ── Phase 2: refit a fresh model on train+val for the selected epoch count ────
if args.refit_trainval:
    refit_epochs = args.refit_epochs if args.refit_epochs > 0 else best_epoch
    print(f"\n{'='*60}")
    print(f"Refitting on train+val for {refit_epochs} epochs")
    print(f"{'='*60}")
    print(f"  samples {len(x_train)} -> {len(x_fit)}   "
          f"steps/epoch {int(np.ceil(len(x_train)/args.batch_size))} -> "
          f"{int(np.ceil(len(x_fit)/args.batch_size))}")

    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, refit_epochs + 1):
        # Replay the phase-1 schedule; hold the last rate if we run past it.
        lr = lr_history[min(epoch, len(lr_history)) - 1]
        for g in optimizer.param_groups:
            g["lr"] = lr
        t0 = time.time()
        tr_mae = train_one_epoch(model, optimizer, x_fit, y_fit, fit_mean, fit_std)
        if epoch % 20 == 0 or epoch == refit_epochs:
            print(f"{epoch:6d}  {tr_mae:11.4f}  {'':11}  {'':11}"
                  f"  {lr:8.2e}  {time.time()-t0:5.1f}s", flush=True)

    torch.save(model.state_dict(), ckpt_path)
else:
    model.load_state_dict(torch.load(sel_ckpt_path, map_location=device))
    torch.save(model.state_dict(), ckpt_path)

# ── Test evaluation ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Test evaluation (fit on {FIT_ON})")
print(f"{'='*60}")
model.eval()

all_pred, all_real = [], []
with torch.no_grad():
    for xb, yb in make_batches(x_test, y_test, args.batch_size):
        pred, real = forward_eval(model, xb, yb, fit_mean, fit_std)   # raw scale
        all_pred.append(pred.cpu())
        all_real.append(real.cpu())

preds_raw = torch.cat(all_pred, dim=0).numpy()   # (S_test, H, N, F)
reals_raw = torch.cat(all_real, dim=0).numpy()   # (S_test, H, N, F)

smean, sstd = load_scaler(data_dir, FIT_ON)      # (N, F) each

# ── Metrics via shared metrics.py ─────────────────────────────────────────────
results = evaluate(preds_raw, reals_raw, smean, sstd, ns,
                   horizons=eval_horizons)

print_results_table(
    {"MTGNN": results},
    detail_h=min(12, max(eval_horizons)),
    horizons=eval_horizons,
    title="Table: Forecasting Performance",
)

# ── Detailed per-feature breakdown ────────────────────────────────────────────
preds_t = torch.from_numpy(preds_raw)
reals_t = torch.from_numpy(reals_raw)
print(f"\n{'Feature':<52}  {'raw MAE':>10}  {'raw RMSE':>10}  {'MAPE(%)':>9}")
print("-" * 90)
feat_mae  = torch.mean(torch.abs(preds_t - reals_t), dim=(0, 1, 2))           # (F,)
feat_rmse = torch.sqrt(torch.mean((preds_t - reals_t) ** 2, dim=(0, 1, 2)))   # (F,)
safe_reals = reals_t.clone()
safe_reals[safe_reals.abs() < 1e-8] = float("nan")
feat_mape = torch.nanmean(
    (torch.abs(preds_t - reals_t) / safe_reals.abs() * 100), dim=(0, 1, 2)
)
for fi, fn in enumerate(feature_names):
    mape_val = feat_mape[fi].item()
    flag = " <--" if mape_val > 30.0 else ""
    print(f"  {fn:<50}  {feat_mae[fi].item():>10.4f}  {feat_rmse[fi].item():>10.4f}"
          f"  {mape_val:>9.2f}{flag}")

mean_mape = feat_mape.nanmean().item()
print(f"\n  Mean MAPE (all features) : {mean_mape:.2f}%")
print(f"  (raw MAE in original unit; MAPE in %; MAE/RMSE in table are normalised)")
print(f"\nBest val MAE (normalised)  : {best_val_mae:.4f}")
print(f"Checkpoint                 : {ckpt_path}")

if args.save_metrics:
    import json as _json
    metrics_path = Path(args.save_metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lr": args.lr,
        "seed": args.seed,
        "seq_in": L,
        "seq_out": H,
        "horizon_decoder": args.horizon_decoder,
        "feature_graph": args.feature_graph,
        "best_val_mae": float(best_val_mae),
        "best_epoch": int(best_epoch),
        "fit_on": FIT_ON,
        "test": {str(h): {k: float(v) for k, v in results[h].items()}
                 for h in eval_horizons if h in results},
        "ckpt": str(ckpt_path),
    }
    metrics_path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Metrics JSON               : {metrics_path}")
