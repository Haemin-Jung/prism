"""
eval_baselines.py — VAR, ARIMA, LSTM, TCN baselines for US Industrial Landscape.

All baselines share the same train / val / test splits produced by
prepare_industrial.py and are evaluated with the same five metrics (MAE, RMSE,
MAPE, RSE, MASE) from metrics.py.

Rolling-forecast protocol
──────────────────────────
  Statistical models (VAR, ARIMA) use the full raw sequential time series from
  US_Industrial_Landscape_complete_tensor.npz.

  • Fit on months   0 … T_fit-1              (= n_train + L months)
  • Initial update  T_fit … T_test_start-1    (= n_val months, to align with
                                               the first test window)
  • For each of the S_test test samples: predict H steps, then update with the
    newly observed month — a strict non-leaking expanding-window evaluation.

Neural baselines (LSTM, TCN) use the same sliding-window (x, y) NPZ splits and
are trained with the same loss (normalised-space MAE), optimiser, and early-
stopping patience as MTGNN.  Checkpoints are cached so training is skipped on
subsequent runs.

Usage examples
──────────────
  # run all four baselines (may take 30–90 min on first run)
  python eval_baselines.py

  # only statistical baselines (fast)
  python eval_baselines.py --models var,arima

  # only neural baselines, skip training if checkpoints exist
  python eval_baselines.py --models lstm,tcn

  # save results JSON for merging with MTGNN eval later
  python eval_baselines.py --save results/baselines.json

  # merge saved baseline + MTGNN results and print combined table
  python eval_mtgnn.py --n_mc 30 --merge results/baselines.json
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from metrics import evaluate, naive_scale, print_results_table, save_results_json, save_results_excel
from data_utils import fit_targets, load_fit_split, load_meta, load_scaler, load_split

warnings.filterwarnings("ignore")

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",   default="data/industrial_L12_H24")
parser.add_argument("--raw_npz",    default="US_Industrial_Landscape_complete_tensor.npz",
                    help="full sequential tensor; required for VAR and ARIMA")
parser.add_argument("--models",     default="naive_last,naive_seasonal,var,arima,lstm,tcn,patchtst,lightgbm",
                    help="comma-separated list of models to run  "
                         "(naive_last, naive_seasonal, var, arima, lstm, tcn, "
                         "patchtst, lightgbm)")
parser.add_argument("--device",     default="cuda:0")
parser.add_argument("--horizons",   default="3,6,12,24")
parser.add_argument("--detail_h",   type=int, default=0,
                    help="horizon for the detailed five-metric columns (0 = auto-detect best)")
parser.add_argument("--save",       default="results/baselines.json",
                    help="path to write results JSON")
parser.add_argument("--save_excel", default="",
                    help="also write results to an Excel file (.xlsx)")
# LSTM / TCN training hyper-parameters
parser.add_argument("--epochs",     type=int, default=200)
parser.add_argument("--batch",      type=int, default=16)
parser.add_argument("--lr",         type=float, default=1e-3)
parser.add_argument("--patience",   type=int, default=60,
                    help="early-stopping patience; must leave room for several "
                         "--lr_patience cycles or runs die before a reduced "
                         "learning rate can take effect")
parser.add_argument("--lr_patience", type=int, default=10,
                    help="ReduceLROnPlateau patience in epochs")
parser.add_argument("--min_epochs", type=int, default=30,
                    help="never early-stop before this epoch")
parser.add_argument("--seed",       type=int, default=42,
                    help="seed for LSTM/TCN init and shuffling; the checkpoint name "
                         "carries it so separate seeds do not overwrite each other")
parser.add_argument("--hidden",     type=int, default=256,  help="LSTM hidden size")
parser.add_argument("--tcn_ch",     type=int, default=128,  help="TCN channel width")
parser.add_argument("--patch_len",  type=int, default=4,    help="PatchTST patch length")
parser.add_argument("--patch_stride", type=int, default=2,  help="PatchTST stride")
parser.add_argument("--patch_d_model", type=int, default=64, help="PatchTST d_model")
parser.add_argument("--lgb_estimators", type=int, default=200,
                    help="LightGBM n_estimators (per horizon head)")
parser.add_argument("--ckpt_dir",   default="checkpoints/industrial")
parser.add_argument("--refit_trainval", action="store_true",
                    help="select the epoch count on val, then refit LSTM/TCN on "
                         "train+val before testing. Also switches the scaler and "
                         "the MASE denominator to train+val for every model here, "
                         "so it must match the flag used by train_industrial.py.")
args = parser.parse_args()

CKPT_SUFFIX = ("_trval" if args.refit_trainval else "") + f"_s{args.seed}"

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.device.startswith("cuda") and not torch.cuda.is_available():
    args.device = "cpu"
device   = torch.device(args.device)
horizons = [int(h) for h in args.horizons.split(",")]
models   = [m.strip().lower() for m in args.models.split(",")]

# ── Data loading ──────────────────────────────────────────────────────────────
data_dir = Path(args.data_dir)
meta     = load_meta(data_dir)
L, H     = meta["seq_in"], meta["seq_out"]
N, F_dim = meta["n_nodes"], meta["n_features"]
n_train  = meta["n_train"]
n_val    = meta["n_val"]
S_test   = meta["n_test"]

# Must match what train_industrial.py used, otherwise the reported MAE/RMSE
# (normalised space) and MASE (naive denominator) are not comparable.
FIT_ON = "trainval" if args.refit_trainval else "train"


def _t(a):
    return torch.tensor(a, dtype=torch.float32)


# Epoch-count selection runs under the train-only regime …
sel_mean, sel_std = load_scaler(data_dir, "train")
x_train, y_train  = map(_t, load_split(data_dir, "train", "train"))
x_val,   y_val    = map(_t, load_split(data_dir, "val",   "train"))

# … while the final fit and every reported metric follow FIT_ON.
smean, sstd  = load_scaler(data_dir, FIT_ON)
x_fit, y_fit = map(_t, load_fit_split(data_dir, FIT_ON))
_x_test, y_test = load_split(data_dir, "test", FIT_ON)
x_test = _t(_x_test)

ns = naive_scale(fit_targets(data_dir, FIT_ON))               # (N, F)

print(f"Data: N={N}, F={F_dim}, L={L}, H={H}")
print(f"Splits: train={len(x_train)}, val={len(x_val)}, test={S_test}")
print(f"Fit on: {FIT_ON}  ({len(x_fit)} samples)")
print(f"Device: {device}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_batches(x, y, batch_size, shuffle=False):
    idx = torch.randperm(len(x)) if shuffle else torch.arange(len(x))
    for i in range(0, len(x), batch_size):
        b = idx[i : i + batch_size]
        yield x[b], y[b]


def _run_epoch(model, opt, x, y, s_mean, s_std, train: bool):
    model.train() if train else model.eval()
    losses = []
    with torch.set_grad_enabled(train):
        for xb, yb in make_batches(x, y, args.batch, shuffle=train):
            xb = xb.to(device)
            yb_n = (yb.to(device) - s_mean) / s_std
            pred = model(xb).view(xb.size(0), H, N, F_dim)
            loss = torch.mean(torch.abs(pred - yb_n))
            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            losses.append(loss.item())
    return float(np.mean(losses))


def train_nn(make_model, name: str, ckpt_path: Path):
    """
    Fit a neural baseline under the same protocol as MTGNN.

    Phase 1 trains on `train` and early-stops on `val` to select the epoch
    count.  With --refit_trainval, phase 2 then trains a fresh model on
    train+val for that many epochs, replaying the phase-1 learning-rate
    trajectory since no validation split is left to drive the scheduler.
    """
    model = make_model().to(device)
    if ckpt_path.exists():
        print(f"  [{name}] checkpoint found - skipping training: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return model

    print(f"  [{name}] selecting epoch count (up to {args.epochs} epochs) …")
    opt   = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=args.lr_patience, factor=0.5, min_lr=1e-5
    )
    sel_mean_t = torch.tensor(sel_mean, device=device)
    sel_std_t  = torch.tensor(sel_std,  device=device)

    sel_ckpt = ckpt_path.with_name(ckpt_path.stem + "_selection.pth")
    best_val, best_epoch, patience_cnt = float("inf"), 0, 0
    lr_history = []

    for epoch in range(1, args.epochs + 1):
        lr_history.append(opt.param_groups[0]["lr"])
        tr = _run_epoch(model, opt, x_train, y_train, sel_mean_t, sel_std_t, True)
        vl = _run_epoch(model, opt, x_val,   y_val,   sel_mean_t, sel_std_t, False)
        sched.step(vl)

        if vl < best_val:
            best_val, best_epoch, patience_cnt = vl, epoch, 0
            torch.save(model.state_dict(), sel_ckpt)
        else:
            patience_cnt += 1

        if epoch % 20 == 0 or patience_cnt == 0:
            print(f"    ep {epoch:4d}  tr={tr:.4f}  val={vl:.4f}  best={best_val:.4f}"
                  + (" *" if patience_cnt == 0 else ""))

        if patience_cnt >= args.patience and epoch >= args.min_epochs:
            print(f"    early stop at epoch {epoch}")
            break

    print(f"  [{name}] best val MAE = {best_val:.4f}  at epoch {best_epoch}")

    if not args.refit_trainval:
        model.load_state_dict(torch.load(sel_ckpt, map_location=device))
        torch.save(model.state_dict(), ckpt_path)
        return model

    print(f"  [{name}] refitting on train+val for {best_epoch} epochs "
          f"({len(x_train)} -> {len(x_fit)} samples) …")
    model = make_model().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    fit_mean_t = torch.tensor(smean, device=device)
    fit_std_t  = torch.tensor(sstd,  device=device)

    for epoch in range(1, best_epoch + 1):
        lr = lr_history[min(epoch, len(lr_history)) - 1]
        for g in opt.param_groups:
            g["lr"] = lr
        tr = _run_epoch(model, opt, x_fit, y_fit, fit_mean_t, fit_std_t, True)
        if epoch % 20 == 0 or epoch == best_epoch:
            print(f"    ep {epoch:4d}  tr={tr:.4f}  lr={lr:.2e}")

    torch.save(model.state_dict(), ckpt_path)
    return model


def eval_nn(model, name: str) -> np.ndarray:
    """Run test-set inference; returns (S, H, N, F) raw-scale predictions."""
    model.eval()
    smean_t = torch.tensor(smean, device=device)
    sstd_t  = torch.tensor(sstd,  device=device)
    preds = []
    with torch.no_grad():
        for xb, _ in make_batches(x_test, torch.zeros(len(x_test)), args.batch):
            xb   = xb.to(device)
            pred = model(xb).view(xb.size(0), H, N, F_dim)
            preds.append((pred * sstd_t + smean_t).cpu().numpy())
    return np.concatenate(preds, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Neural baseline models
# ══════════════════════════════════════════════════════════════════════════════

class LSTMForecaster(nn.Module):
    """
    2-layer LSTM → MLP projection.
    Input : (B, L, N*F)  z-score normalised
    Output: (B, H*N*F)   z-score normalised  (reshaped to (B,H,N,F) outside)
    """
    def __init__(self, in_dim, hidden, n_layers, out_dim, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, n_layers,
                            batch_first=True, dropout=dropout)
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        # x : (B, L, N, F) → flatten N×F → (B, L, N*F)
        B = x.size(0)
        x = x.reshape(B, L, N * F_dim)
        _, (h, _) = self.lstm(x)
        return self.proj(h[-1])   # last layer's final hidden state


class _TCNBlock(nn.Module):
    """One residual dilated-conv block (causal padding, weight-normalised)."""
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout=0.1):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad))
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation, padding=pad))
        self.drop  = nn.Dropout(dropout)
        self.relu  = nn.ReLU()
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        # causal trimming: remove right-padding to keep the time dim constant
        o = self.relu(self.drop(self.conv1(x)[..., : x.size(-1)]))
        o = self.relu(self.drop(self.conv2(o)[..., : x.size(-1)]))
        res = x if self.skip is None else self.skip(x)
        return self.relu(o + res)


class TCNForecaster(nn.Module):
    """
    4-block dilated TCN (dilations 1,2,4,8 → receptive field ≥ 36).
    Input : (B, L, N*F)  z-score normalised
    Output: (B, H*N*F)   z-score normalised
    """
    def __init__(self, in_dim, channels, kernel, out_dim, dropout=0.1):
        super().__init__()
        blocks, prev = [], in_dim
        for i, ch in enumerate(channels):
            blocks.append(_TCNBlock(prev, ch, kernel, 2 ** i, dropout))
            prev = ch
        self.net  = nn.Sequential(*blocks)
        self.proj = nn.Linear(channels[-1], out_dim)

    def forward(self, x):
        B = x.size(0)
        # (B, L, N*F) → (B, N*F, L)  for Conv1d
        x = x.reshape(B, L, N * F_dim).permute(0, 2, 1)
        o = self.net(x)[:, :, -1]     # last time-step features
        return self.proj(o)


class PatchTSTForecaster(nn.Module):
    """
    Compact channel-independent PatchTST-style forecaster.

    Input : (B, L, N, F)  → treat C=N*F channels independently
    Output: (B, H*N*F)    normalised
    """

    def __init__(self, seq_len, n_channels, pred_len, patch_len=4, stride=2,
                 d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1
        if self.n_patches < 1:
            raise ValueError(
                f"PatchTST: seq_len={seq_len} too short for "
                f"patch_len={patch_len}, stride={stride}"
            )
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.n_patches * d_model, pred_len),
        )
        self.drop = nn.Dropout(dropout)

    def _patchify(self, x):
        # x: (B*C, L) → (B*C, n_patches, patch_len)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return patches

    def forward(self, x):
        B = x.size(0)
        # (B, L, N, F) → (B, C, L)
        x = x.reshape(B, self.seq_len, self.n_channels).permute(0, 2, 1)
        bc = B * self.n_channels
        x = x.reshape(bc, self.seq_len)
        p = self._patchify(x)                          # (B*C, P, patch_len)
        p = self.drop(self.patch_proj(p) + self.pos)   # (B*C, P, d)
        p = self.encoder(p)
        y = self.head(p)                               # (B*C, H)
        y = y.view(B, self.n_channels, self.pred_len).permute(0, 2, 1).contiguous()
        return y.reshape(B, self.pred_len * self.n_channels)


def _lgb_xy(x_t: torch.Tensor, y_t: torch.Tensor | None, s_mean, s_std):
    """Window tensors → numpy features / multi-horizon targets (normalised)."""
    x = x_t.numpy()
    s_mean = np.asarray(s_mean, dtype=np.float32)
    s_std = np.asarray(s_std, dtype=np.float32)
    xn = (x - s_mean) / s_std
    # per-series lags + panel-mean lags (keeps feature dim modest)
    # xn: (S, L, N, F)
    S = xn.shape[0]
    series = xn.reshape(S, L, N * F_dim)                 # (S, L, C)
    panel = series.mean(axis=2, keepdims=True)           # (S, L, 1)
    # build long rows: one per (sample, channel)
    C = N * F_dim
    feats = []
    for c in range(C):
        own = series[:, :, c]                            # (S, L)
        ctx = panel[:, :, 0]                             # (S, L)
        feats.append(np.concatenate([own, ctx], axis=1))  # (S, 2L)
    X = np.concatenate(feats, axis=0).astype(np.float32)  # (S*C, 2L)
    if y_t is None:
        return X, None
    y = y_t.numpy()
    yn = (y - s_mean) / s_std
    Y = yn.reshape(S, H, C).transpose(0, 2, 1).reshape(S * C, H).astype(np.float32)
    return X, Y


def train_lightgbm(ckpt_path: Path):
    """Fit H-output LightGBM on long-format (sample×channel) rows."""
    import joblib
    from lightgbm import LGBMRegressor
    from sklearn.multioutput import MultiOutputRegressor

    if ckpt_path.exists():
        print(f"  [LightGBM] checkpoint found - skipping training: {ckpt_path}")
        return joblib.load(ckpt_path)

    X_tr, Y_tr = _lgb_xy(x_train, y_train, sel_mean, sel_std)
    X_vl, Y_vl = _lgb_xy(x_val, y_val, sel_mean, sel_std)

    base = LGBMRegressor(
        n_estimators=args.lgb_estimators,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model = MultiOutputRegressor(base, n_jobs=1)

    print(f"  [LightGBM] fitting MultiOutput(H={H}) on {X_tr.shape[0]} rows "
          f"× {X_tr.shape[1]} feats …")
    # Early-ish selection: fit on train, score val MAE, optionally shrink trees
    # via a second fit on train+val with same n_estimators (refit protocol).
    model.fit(X_tr, Y_tr)
    pred_vl = model.predict(X_vl)
    val_mae = float(np.mean(np.abs(pred_vl - Y_vl)))
    print(f"  [LightGBM] val MAE (norm) = {val_mae:.4f}")

    if args.refit_trainval:
        X_fit_np, Y_fit_np = _lgb_xy(x_fit, y_fit, smean, sstd)
        print(f"  [LightGBM] refitting on train+val ({X_fit_np.shape[0]} rows) …")
        model = MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=args.lgb_estimators,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=args.seed,
                n_jobs=-1,
                verbosity=-1,
            ),
            n_jobs=1,
        )
        model.fit(X_fit_np, Y_fit_np)

    joblib.dump(model, ckpt_path)
    return model


def eval_lightgbm(model) -> np.ndarray:
    """Return (S, H, N, F) raw-scale predictions."""
    X_te, _ = _lgb_xy(x_test, None, smean, sstd)
    pred = model.predict(X_te).astype(np.float32)   # (S*C, H)
    S = len(x_test)
    C = N * F_dim
    pred = pred.reshape(S, C, H).transpose(0, 2, 1).reshape(S, H, N, F_dim)
    return pred * sstd + smean


# ══════════════════════════════════════════════════════════════════════════════
# Statistical baselines — Naive, VAR and ARIMA
# ══════════════════════════════════════════════════════════════════════════════

def _load_full_series() -> np.ndarray:
    """
    Return the full raw sequential time series, shape (T, N, F).
    Raises FileNotFoundError if the raw NPZ is missing.
    """
    raw_path = Path(args.raw_npz)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw NPZ not found: {raw_path}\n"
            "VAR and ARIMA require the full sequential series.\n"
            "Run build_US_Industrial_Landscape_full.py first."
        )
    d = np.load(raw_path)
    return d["data"].transpose(2, 0, 1).astype(np.float64)   # (T, N, F)


def _rolling_context(full_series: np.ndarray, T_test_start: int):
    """
    Generator: yield (s, context_end_idx) for each test sample.
    context_end_idx is the last observed month index before the h=1 prediction.
    """
    for s in range(S_test):
        yield s, T_test_start + s   # first prediction = month T_test_start + s


# ── Naive baselines ────────────────────────────────────────────────────────────
def run_naive_last(full_series: np.ndarray) -> np.ndarray:
    """
    Last-value naive: ŷ_{t+h} = y_t  for all h = 1 … H
    Simply repeats the last observed value across the entire horizon.
    Returns (S_test, H, N, F).
    """
    T_test_start = n_train + n_val + L
    preds = np.zeros((S_test, H, N, F_dim), dtype=np.float32)
    for s in range(S_test):
        last = full_series[T_test_start + s - 1].astype(np.float32)  # (N, F)
        preds[s] = last[np.newaxis, :, :]   # broadcast to (H, N, F)
    return preds


def run_naive_seasonal(full_series: np.ndarray, m: int = 12) -> np.ndarray:
    """
    Seasonal naive: ŷ_{t+h} = y_{t+h−m·⌈h/m⌉}  for h = 1 … H

    The number of seasonal cycles stepped back grows with the horizon, so the
    reference month never sits past the forecast origin.  Taking a single step
    back (y_{t+h−m}) would read unobserved future months for every h > m.
    Returns (S_test, H, N, F).
    """
    T_test_start = n_train + n_val + L
    preds = np.zeros((S_test, H, N, F_dim), dtype=np.float32)
    for s in range(S_test):
        for h in range(H):
            cycles  = -(-(h + 1) // m)                         # ceil((h+1)/m)
            src_idx = T_test_start + s + h - m * cycles
            assert src_idx < T_test_start + s, "seasonal naive read a future month"
            preds[s, h] = full_series[max(src_idx, 0)].astype(np.float32)
    return preds


# ── VAR ───────────────────────────────────────────────────────────────────────
def run_var(full_series: np.ndarray) -> np.ndarray:
    """
    Node-wise VAR(p) — one 10-variable VAR per industry node.

    Fit on months 0 … T_fit-1.  For each test sample the forecast context
    is the lag-order window ending at the start of that test prediction.
    Returns (S_test, H, N, F) raw predictions.
    """
    from statsmodels.tsa.vector_ar.var_model import VAR

    T_fit        = n_train + L          # last training month (exclusive)
    T_test_start = n_train + n_val + L  # first test prediction month

    preds = np.zeros((S_test, H, N, F_dim), dtype=np.float32)

    for n_idx in range(N):
        node_train = full_series[:T_fit, n_idx, :]   # (T_fit, F)
        # statsmodels requires: maxlags < T / (k_vars + 1)
        # keep headroom for AIC selection (the largest candidate model must be estimable)
        safe_maxlags = max(1, min(12, T_fit // (F_dim + 1) - 2))
        model_res  = VAR(node_train).fit(maxlags=safe_maxlags, ic="aic", trend="c")
        lag = model_res.k_ar

        for s in range(S_test):
            # context: the lag-order months ending just before the prediction
            ctx_end   = T_test_start + s
            ctx_start = ctx_end - lag
            context   = full_series[ctx_start:ctx_end, n_idx, :]   # (lag, F)
            preds[s, :, n_idx, :] = model_res.forecast(context, steps=H)

    return preds


# ── ARIMA ─────────────────────────────────────────────────────────────────────
def run_arima(full_series: np.ndarray) -> np.ndarray:
    """
    Per-(node, feature) auto-ARIMA with seasonal component (m=12).

    Rolling evaluation:
      • Fit on months 0 … T_fit-1
      • Update with months T_fit … T_test_start-1  (= n_val months)
      • For s=0..S_test-1: predict H, then update with month T_test_start+s
    Returns (S_test, H, N, F) raw predictions.
    """
    from pmdarima import auto_arima as _auto

    T_fit        = n_train + L
    T_test_start = n_train + n_val + L

    preds = np.zeros((S_test, H, N, F_dim), dtype=np.float32)
    total = N * F_dim
    done  = 0

    for n_idx in range(N):
        for f_idx in range(F_dim):
            t0  = time.time()
            ser = full_series[:, n_idx, f_idx]

            model = _auto(
                ser[:T_fit],
                seasonal=True, m=12,
                max_p=3, max_q=3, max_P=1, max_Q=1,
                stepwise=True, suppress_warnings=True,
                error_action="ignore",
            )

            # bring ARIMA up to the test start
            init_obs = ser[T_fit:T_test_start]
            if len(init_obs) > 0:
                model.update(init_obs)

            # rolling forecast over S_test windows
            for s in range(S_test):
                preds[s, :, n_idx, f_idx] = model.predict(n_periods=H)
                if s < S_test - 1:
                    model.update([ser[T_test_start + s]])

            done += 1
            elapsed = time.time() - t0
            if done % 10 == 0:
                print(f"    ARIMA fitted {done}/{total}  "
                      f"(last: node={n_idx} feat={f_idx}, {elapsed:.1f}s)")

    return preds


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
all_results = {}
ckpt_dir    = Path(args.ckpt_dir)
ckpt_dir.mkdir(parents=True, exist_ok=True)
full_series = None   # loaded on first use by any statistical baseline

# ── Naive baselines (need full_series, but trivially fast) ────────────────────
need_series = any(m in models for m in ("var", "arima", "naive_last", "naive_seasonal"))
if need_series:
    full_series = _load_full_series()

if "naive_last" in models:
    print("=" * 60)
    print("Running Naive-Last …")
    t0 = time.time()
    nl_preds = run_naive_last(full_series)
    print(f"  Naive-Last done in {time.time()-t0:.1f}s")
    all_results["Naive-Last"] = evaluate(nl_preds, y_test, smean, sstd, ns, horizons)

if "naive_seasonal" in models:
    print("=" * 60)
    print("Running Naive-Seasonal (m=12) …")
    t0 = time.time()
    ns_preds = run_naive_seasonal(full_series)
    print(f"  Naive-Seasonal done in {time.time()-t0:.1f}s")
    all_results["Naive-Seasonal"] = evaluate(ns_preds, y_test, smean, sstd, ns, horizons)

# ── VAR ───────────────────────────────────────────────────────────────────────
if "var" in models:
    print("=" * 60)
    print("Running VAR …")
    t0 = time.time()
    var_preds   = run_var(full_series)
    print(f"  VAR done in {time.time()-t0:.1f}s")
    all_results["VAR"] = evaluate(var_preds, y_test, smean, sstd, ns, horizons)

# ── ARIMA ─────────────────────────────────────────────────────────────────────
if "arima" in models:
    print("=" * 60)
    print("Running ARIMA (auto_arima per series, may take 15-40 min) ...")
    t0 = time.time()
    arima_preds = run_arima(full_series)
    print(f"  ARIMA done in {time.time()-t0:.1f}s")
    all_results["ARIMA"] = evaluate(arima_preds, y_test, smean, sstd, ns, horizons)

# ── LSTM ──────────────────────────────────────────────────────────────────────
if "lstm" in models:
    print("=" * 60)
    print("Running LSTM …")
    t0        = time.time()
    lstm_ckpt = ckpt_dir / f"lstm_L{L}_H{H}{CKPT_SUFFIX}.pth"

    def make_lstm():
        torch.manual_seed(args.seed)
        return LSTMForecaster(
            in_dim   = N * F_dim,
            hidden   = args.hidden,
            n_layers = 2,
            out_dim  = H * N * F_dim,
            dropout  = 0.1,
        )

    lstm = train_nn(make_lstm, "LSTM", lstm_ckpt)
    lstm_preds = eval_nn(lstm, "LSTM")
    print(f"  LSTM done in {time.time()-t0:.1f}s")
    all_results["LSTM"] = evaluate(lstm_preds, y_test, smean, sstd, ns, horizons)

# ── TCN ───────────────────────────────────────────────────────────────────────
if "tcn" in models:
    print("=" * 60)
    print("Running TCN …")
    t0       = time.time()
    tcn_ckpt = ckpt_dir / f"tcn_L{L}_H{H}{CKPT_SUFFIX}.pth"
    ch       = args.tcn_ch

    def make_tcn():
        torch.manual_seed(args.seed)
        return TCNForecaster(
            in_dim   = N * F_dim,
            channels = [ch, ch, ch * 2, ch * 2],   # dilations 1,2,4,8 → RF=121>36
            kernel   = 3,
            out_dim  = H * N * F_dim,
            dropout  = 0.1,
        )

    tcn = train_nn(make_tcn, "TCN", tcn_ckpt)
    tcn_preds = eval_nn(tcn, "TCN")
    print(f"  TCN done in {time.time()-t0:.1f}s")
    all_results["TCN"] = evaluate(tcn_preds, y_test, smean, sstd, ns, horizons)

# ── PatchTST ──────────────────────────────────────────────────────────────────
if "patchtst" in models:
    print("=" * 60)
    print("Running PatchTST …")
    t0 = time.time()
    pt_ckpt = ckpt_dir / f"patchtst_L{L}_H{H}{CKPT_SUFFIX}.pth"

    def make_patchtst():
        torch.manual_seed(args.seed)
        return PatchTSTForecaster(
            seq_len=L,
            n_channels=N * F_dim,
            pred_len=H,
            patch_len=min(args.patch_len, L),
            stride=min(args.patch_stride, max(1, min(args.patch_len, L))),
            d_model=args.patch_d_model,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
        )

    pt = train_nn(make_patchtst, "PatchTST", pt_ckpt)
    pt_preds = eval_nn(pt, "PatchTST")
    print(f"  PatchTST done in {time.time()-t0:.1f}s")
    all_results["PatchTST"] = evaluate(pt_preds, y_test, smean, sstd, ns, horizons)

# ── LightGBM ──────────────────────────────────────────────────────────────────
if "lightgbm" in models:
    print("=" * 60)
    print("Running LightGBM …")
    t0 = time.time()
    lgb_ckpt = ckpt_dir / f"lightgbm_L{L}_H{H}{CKPT_SUFFIX}.joblib"
    lgb = train_lightgbm(lgb_ckpt)
    lgb_preds = eval_lightgbm(lgb)
    print(f"  LightGBM done in {time.time()-t0:.1f}s")
    all_results["LightGBM"] = evaluate(lgb_preds, y_test, smean, sstd, ns, horizons)

# ── Results ───────────────────────────────────────────────────────────────────
print_results_table(
    all_results,
    detail_h = args.detail_h if args.detail_h > 0 else None,
    horizons = horizons,
    title    = "Baseline Forecasting Performance Comparison",
)

if args.save:
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    save_results_json(all_results, args.save, fit_on=FIT_ON)

if args.save_excel:
    Path(args.save_excel).parent.mkdir(parents=True, exist_ok=True)
    save_results_excel(all_results, args.save_excel, horizons=horizons)
