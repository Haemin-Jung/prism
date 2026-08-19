"""
Reproduce the L=12 H=24 comparison: MTGNN variants vs baselines.

For each seed 1..5:
  1. Train HGRU-CTX+HE and HGRU-CTX+HE + FGraph (skip if checkpoint exists)
  2. Evaluate Naive / ARIMA / VAR / LSTM / TCN / PatchTST / LightGBM
  3. Evaluate MTGNN, MTGNN+FGraph, and B-MTGNN+FGraph (MC-Dropout)
  4. Write results/seed{s}_L12_H24.json

Then aggregate with summarize.py.

Usage
    python run_experiment.py --device cuda:0
    python run_experiment.py --skip_train
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--python", default=sys.executable)
parser.add_argument("--seeds", default="1,2,3,4,5")
parser.add_argument("--n_mc", type=int, default=30)
parser.add_argument("--lr", type=float, default=2e-3)
parser.add_argument("--skip_prepare", action="store_true")
parser.add_argument("--skip_train", action="store_true",
                    help="evaluate existing checkpoints only")
parser.add_argument("--skip_baselines", action="store_true",
                    help="reuse models already stored in seed JSON files")
args = parser.parse_args()

root = Path(__file__).resolve().parent
py = args.python
seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
L, H = 12, 24
data = root / f"data/industrial_L{L}_H{H}"
ckpt_dir = root / "checkpoints" / "industrial"
results = root / "results"
results.mkdir(parents=True, exist_ok=True)
hors = "3,6,12,24"
ARCH = [
    "--layers", "3", "--dilation_exp", "1",
    "--conv_ch", "32", "--residual_ch", "32",
    "--skip_ch", "64", "--end_ch", "128",
    "--horizon_decoder", "gru_ctx_hemb",
    "--horizon_emb_dim", "8",
]


def run(cmd: list[str]) -> None:
    print(">", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd, cwd=str(root),
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"},
    )
    if proc.returncode != 0:
        raise SystemExit(f"failed ({proc.returncode}): {' '.join(cmd)}")


def merge_json(dst: Path, *srcs: Path) -> None:
    merged: dict = {}
    proto = {"fit_on": "trainval"}
    for src in srcs:
        if not src.exists():
            continue
        d = json.loads(src.read_text(encoding="utf-8"))
        proto = d.get("_protocol") or proto
        for k, v in d.items():
            if str(k).startswith("_"):
                continue
            merged[k] = v
    merged["_protocol"] = proto
    dst.write_text(json.dumps(merged, indent=2), encoding="utf-8")


def train_mtgnn(expid: str, feature_graph: bool, seed: int) -> Path:
    ckpt = ckpt_dir / f"{expid}.pth"
    if args.skip_train or ckpt.exists():
        if not ckpt.exists():
            raise SystemExit(f"missing checkpoint {ckpt}")
        print(f"  skip train: {ckpt}", flush=True)
        return ckpt
    cmd = [
        py, "-u", str(root / "train_industrial.py"),
        "--data_dir", str(data),
        "--seq_in", str(L), "--seq_out", str(H),
        "--device", args.device,
        "--refit_trainval",
        "--seed", str(seed),
        "--expid", expid,
        "--lr", str(args.lr),
        "--loss", "mae",
        *ARCH,
    ]
    if feature_graph:
        cmd.append("--feature_graph")
    run(cmd)
    return ckpt


def eval_mtgnn(ckpt: Path, seed_json: Path, tmp: Path, *,
               feature_graph: bool, n_mc: int, both: bool,
               label_suffix: str) -> None:
    cmd = [
        py, "-u", str(root / "eval_mtgnn.py"),
        "--ckpt", str(ckpt),
        "--data_dir", str(data),
        "--fit_on", "trainval",
        "--device", args.device,
        "--n_mc", str(n_mc),
        "--horizons", hors,
        "--detail_h", str(H),
        *ARCH,
        "--label_suffix", label_suffix,
        "--merge", str(seed_json),
        "--save", str(tmp),
    ]
    if feature_graph:
        cmd.append("--feature_graph")
    if both:
        cmd.append("--both")
    run(cmd)
    seed_json.write_text(tmp.read_text(encoding="utf-8"), encoding="utf-8")
    tmp.unlink(missing_ok=True)


if not args.skip_prepare:
    run([
        py, "-u", str(root / "prepare_industrial.py"),
        "--seq_in", str(L), "--seq_out", str(H),
    ])

for s in seeds:
    print(f"\n########## seed {s} ##########", flush=True)
    seed_json = results / f"seed{s}_L{L}_H{H}.json"
    ckpt_nf = train_mtgnn(f"mtgnn_s{s}_L{L}_H{H}", False, s)
    ckpt_fg = train_mtgnn(f"mtgnn_fgraph_s{s}_L{L}_H{H}", True, s)

    if not args.skip_baselines:
        tmp_b = results / f"_tmp_base_s{s}_L{L}_H{H}.json"
        run([
            py, "-u", str(root / "eval_baselines.py"),
            "--data_dir", str(data),
            "--device", args.device,
            "--refit_trainval",
            "--seed", str(s),
            "--models", "naive_last,naive_seasonal,var,arima,lstm,tcn,patchtst,lightgbm",
            "--horizons", hors,
            "--save", str(tmp_b),
        ])
        merge_json(seed_json, seed_json, tmp_b)
        tmp_b.unlink(missing_ok=True)
    elif not seed_json.exists():
        raise SystemExit(f"--skip_baselines but missing {seed_json}")

    tmp = results / f"_tmp_mtgnn_s{s}_L{L}_H{H}.json"
    eval_mtgnn(
        ckpt_nf, seed_json, tmp,
        feature_graph=False, n_mc=0, both=False,
        label_suffix=" HGRU-CTX+HE",
    )
    eval_mtgnn(
        ckpt_fg, seed_json, tmp,
        feature_graph=True, n_mc=args.n_mc, both=True,
        label_suffix=" HGRU-CTX+HE FGraph",
    )
    print(f"  wrote {seed_json}", flush=True)

run([py, "-u", str(root / "summarize.py")])
print("\nDone. Per-seed JSON: results/seed*_L12_H24.json", flush=True)
print("Summary          : results/summary_L12_H24.json", flush=True)
