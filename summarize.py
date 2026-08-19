"""Aggregate L=12 H=24 per-seed JSONs into results/summary_L12_H24.json."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

files = sorted(Path("results").glob("seed*_L12_H24.json"))
print("n_files", len(files))
models = [
    "Naive-Last",
    "Naive-Seasonal",
    "ARIMA",
    "VAR",
    "LSTM",
    "TCN",
    "PatchTST",
    "LightGBM",
    "MTGNN HGRU-CTX+HE",
    "MTGNN HGRU-CTX+HE FGraph",
    "B-MTGNN HGRU-CTX+HE FGraph (MC=30)",
]
hs = [3, 6, 12, 24]
bag: dict = defaultdict(lambda: defaultdict(list))
for f in files:
    d = json.loads(f.read_text(encoding="utf-8"))
    for m in models:
        if m not in d:
            print("missing", m, "in", f.name)
            continue
        for h in hs:
            b = d[m].get(str(h))
            if not b:
                continue
            bag[m][("RMSE", h)].append(b["RMSE"])
            bag[m][("MAPE", h)].append(b["MAPE"])


def cell(vals, digits=3):
    a = np.asarray(vals, dtype=float)
    if len(a) == 0:
        return "—"
    if len(a) == 1:
        return f"{a.mean():.{digits}f}"
    return f"{a.mean():.{digits}f}+-{a.std(ddof=1):.{digits}f}"


summary = {"L": 12, "H": 24, "n_seeds": len(files), "horizons": hs, "models": {}}
print("\nRMSE (norm, 5-seed mean+-std; stat baselines n=1)")
hdr = f"{'Model':36s}" + "".join(f"{h:>14d}" for h in hs)
print(hdr)
for m in models:
    cells = [cell(bag[m][("RMSE", h)]) for h in hs]
    print(f"{m:36s}" + "".join(f"{c:>14s}" for c in cells))
    summary["models"][m] = {
        "RMSE": {
            str(h): {
                "mean": float(np.mean(bag[m][("RMSE", h)])) if bag[m][("RMSE", h)] else None,
                "std": float(np.std(bag[m][("RMSE", h)], ddof=1))
                if len(bag[m][("RMSE", h)]) > 1
                else 0.0,
            }
            for h in hs
        },
        "MAPE": {
            str(h): {
                "mean": float(np.mean(bag[m][("MAPE", h)])) if bag[m][("MAPE", h)] else None,
                "std": float(np.std(bag[m][("MAPE", h)], ddof=1))
                if len(bag[m][("MAPE", h)]) > 1
                else 0.0,
            }
            for h in hs
        },
    }

print("\nMAPE (raw decimal)")
print(hdr)
for m in models:
    cells = [cell(bag[m][("MAPE", h)], digits=4) for h in hs]
    print(f"{m:36s}" + "".join(f"{c:>14s}" for c in cells))

out = Path("results/summary_L12_H24.json")
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\nwrote", out)

print("\nBest per horizon (RMSE, skip VAR if >5):")
for h in hs:
    cands = []
    for m in models:
        vals = bag[m][("RMSE", h)]
        if not vals:
            continue
        mu = float(np.mean(vals))
        if m == "VAR" and mu > 5:
            continue
        cands.append((mu, m))
    cands.sort()
    print(f"  h={h}: " + " < ".join(f"{m}({v:.3f})" for v, m in cands[:4]))
