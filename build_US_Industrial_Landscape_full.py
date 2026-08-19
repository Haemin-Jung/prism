#!/usr/bin/env python3
"""
Build the U.S. Industrial Landscape panel:
10 industry nodes x 10 monthly features x T.

Two output variants are produced in parallel:

  _raw      All months from START_YEAR; NaN where source data is absent.
  _complete Only months where every node has fully observed data (no NaN).

Official sources
----------------
BLS CES API: https://api.bls.gov/publicAPI/v2/timeseries/data/
BLS JOLTS metadata: https://download.bls.gov/pub/time.series/jt/jt.series
Census BFS CSV: https://www.census.gov/econ/bfs/csv/bfs_monthly.csv

Outputs (per variant)
---------------------
US_Industrial_Landscape_{tag}_long.csv
US_Industrial_Landscape_{tag}_tensor_wide.csv
US_Industrial_Landscape_{tag}_tensor.npz
US_Industrial_Landscape_{tag}_metadata.json

Dependencies
------------
pip install requests pandas numpy
"""

from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

START_YEAR = 2010
BLS_API = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
JOLTS_SERIES_URL = "https://download.bls.gov/pub/time.series/jt/jt.series"
BFS_URL = "https://www.census.gov/econ/bfs/csv/bfs_monthly.csv"

NODE_CONFIG = [
    ("N01", "Mining and Logging", "10", "110099", ["NAICS21"]),
    ("N02", "Construction", "20", "230000", ["NAICS23"]),
    ("N03", "Manufacturing", "30", "300000", ["NAICSMNF"]),
    ("N04", "Trade, Transportation, and Utilities", "40", "400000",
     ["NAICS22", "NAICS42", "NAICSRET", "NAICSTW"]),
    ("N05", "Information", "50", "510000", ["NAICS51"]),
    ("N06", "Financial Activities", "55", "510099", ["NAICS52", "NAICS53"]),
    ("N07", "Professional and Business Services", "60", "540099",
     ["NAICS54", "NAICS55", "NAICS56"]),
    ("N08", "Private Education and Health Services", "65", "600000",
     ["NAICS61", "NAICS62"]),
    ("N09", "Leisure and Hospitality", "70", "700000",
     ["NAICS71", "NAICS72"]),
    ("N10", "Other Services", "80", "810000", ["NAICS81"]),
]

CES_SUFFIXES = {
    "employment_level_thousands": "01",
    "average_weekly_hours": "02",
    "average_hourly_earnings_nominal": "03",
}
JOLTS_ELEMENTS = {
    "job_openings_rate_pct": "JO",
    "hires_rate_pct": "HI",
    "quits_rate_pct": "QU",
    "layoffs_discharges_rate_pct": "LD",
}
CPI_SERIES = "CUSR0000SA0"


_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


def request_text(url: str, timeout: int = 120) -> str:
    response = requests.get(url, headers=_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(values), size):
        yield values[i:i + size]


def _year_windows(start_year: int, end_year: int, registration_key: str | None) -> List[Tuple[int, int]]:
    """Split [start_year, end_year] into windows that respect the BLS API limit.

    Unregistered users may request at most 10 years per call; registered users 20.
    """
    max_span = 20 if registration_key else 10
    windows: List[Tuple[int, int]] = []
    y = start_year
    while y <= end_year:
        windows.append((y, min(y + max_span - 1, end_year)))
        y += max_span
    return windows


def fetch_bls_series(
    series_ids: Sequence[str],
    start_year: int,
    end_year: int,
    registration_key: str | None = None,
) -> pd.DataFrame:
    """Fetch monthly BLS series, automatically splitting into API-sized windows.

    The BLS public API limits unregistered callers to 10 years and registered
    callers to 20 years per request.  This function breaks the requested range
    into compliant windows and concatenates the results.
    """
    frames: List[pd.DataFrame] = []
    windows = _year_windows(start_year, end_year, registration_key)
    for win_start, win_end in windows:
        print(f"  Fetching BLS {win_start}-{win_end} ...", flush=True)
        for batch in chunks(list(series_ids), 20):
            payload = {
                "seriesid": list(batch),
                "startyear": str(win_start),
                "endyear": str(win_end),
            }
            if registration_key:
                payload["registrationkey"] = registration_key
            response = requests.post(BLS_API, json=payload, timeout=180)
            response.raise_for_status()
            result = response.json()
            if result.get("status") != "REQUEST_SUCCEEDED":
                raise RuntimeError(f"BLS API failure: {result}")
            for series in result["Results"]["series"]:
                rows = []
                for obs in series.get("data", []):
                    period = obs["period"]
                    if not period.startswith("M") or period == "M13":
                        continue
                    rows.append({
                        "date": pd.Timestamp(
                            year=int(obs["year"]),
                            month=int(period[1:]),
                            day=1,
                        ),
                        "series_id": series["seriesID"],
                        "value": pd.to_numeric(obs["value"], errors="coerce"),
                    })
                frames.append(pd.DataFrame(rows))
    if not frames:
        raise RuntimeError("No BLS data returned.")
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["date", "series_id"])


def jolts_series_mapping() -> Dict[Tuple[str, str], str]:
    """Resolve official JOLTS series IDs from metadata rather than guessing IDs."""
    metadata = pd.read_csv(
        io.StringIO(request_text(JOLTS_SERIES_URL)),
        sep="\t",
        dtype=str,
    )
    metadata.columns = [c.strip() for c in metadata.columns]
    for col in metadata.columns:
        metadata[col] = metadata[col].astype(str).str.strip()

    mapping: Dict[Tuple[str, str], str] = {}
    for _, _, _, industry_code, _ in NODE_CONFIG:
        for feature, element in JOLTS_ELEMENTS.items():
            match = metadata[
                (metadata["seasonal"] == "S")
                & (metadata["industry_code"] == industry_code)
                & (metadata["state_code"] == "00")
                & (metadata["area_code"] == "00000")
                & (metadata["sizeclass_code"] == "00")
                & (metadata["dataelement_code"] == element)
                & (metadata["ratelevel_code"] == "R")
            ]
            if len(match) != 1:
                raise RuntimeError(
                    f"Expected one JOLTS series for industry={industry_code}, "
                    f"element={element}; found {len(match)}."
                )
            mapping[(industry_code, feature)] = match.iloc[0]["series_id"]
    return mapping


def load_bfs() -> pd.DataFrame:
    bfs = pd.read_csv(io.StringIO(request_text(BFS_URL)), dtype=str)
    month_cols = ["jan", "feb", "mar", "apr", "may", "jun",
                  "jul", "aug", "sep", "oct", "nov", "dec"]
    bfs = bfs[
        (bfs["sa"] == "A")
        & (bfs["geo"] == "US")
        & (bfs["series"].isin(["BA_BA", "BA_HBA", "BA_WBA"]))
    ].copy()
    long = bfs.melt(
        id_vars=["naics_sector", "series", "year"],
        value_vars=month_cols,
        var_name="month_name",
        value_name="value",
    )
    month_number = {m: i + 1 for i, m in enumerate(month_cols)}
    long["month"] = long["month_name"].map(month_number)
    long["date"] = pd.to_datetime(
        dict(
            year=pd.to_numeric(long["year"]),
            month=long["month"],
            day=1,
        )
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long.dropna(subset=["value"])


SOURCES = {
    "CES_and_JOLTS": BLS_API,
    "JOLTS_metadata": JOLTS_SERIES_URL,
    "BFS": BFS_URL,
}

FEATURE_COLS = [
    "employment_level_thousands",
    "average_weekly_hours",
    "real_average_hourly_earnings_1982_84_dollars",
    "job_openings_rate_pct",
    "hires_rate_pct",
    "quits_rate_pct",
    "layoffs_discharges_rate_pct",
    "business_application_intensity_per_10000_employees",
    "high_propensity_application_share",
    "planned_wage_application_share",
]


def longest_contiguous_run(dates: Sequence) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start, end) of the longest contiguous monthly run in *dates*."""
    dates = sorted(pd.Timestamp(d) for d in dates)
    best = curr = [dates[0]]
    for d in dates[1:]:
        if d == curr[-1] + pd.DateOffset(months=1):
            curr.append(d)
        else:
            if len(curr) > len(best):
                best = curr
            curr = [d]
    if len(curr) > len(best):
        best = curr
    return best[0], best[-1]


def save_outputs(panel: pd.DataFrame, tag: str, out_dir: Path) -> None:
    """Save long CSV, wide CSV, NPZ tensor, and metadata JSON for *panel*."""
    panel = panel.sort_values(["date", "node_id"])
    panel.to_csv(out_dir / f"US_Industrial_Landscape_{tag}_long.csv", index=False)

    wide = panel.pivot(index="date", columns="node_id", values=FEATURE_COLS)
    ordered_cols = [
        (feature, node_id)
        for node_id, *_ in NODE_CONFIG
        for feature in FEATURE_COLS
        if (feature, node_id) in wide.columns
    ]
    wide = wide.loc[:, ordered_cols]
    wide.columns = [f"{node}__{feature}" for feature, node in wide.columns]
    wide.to_csv(out_dir / f"US_Industrial_Landscape_{tag}_tensor_wide.csv")

    dates = sorted(panel["date"].unique())
    node_ids = [x[0] for x in NODE_CONFIG]
    tensor = np.full(
        (len(node_ids), len(FEATURE_COLS), len(dates)), np.nan, dtype=np.float64
    )
    for ni, node_id in enumerate(node_ids):
        node_part = panel[panel["node_id"] == node_id].set_index("date")
        for ti, d in enumerate(dates):
            if d in node_part.index:
                tensor[ni, :, ti] = node_part.loc[d, FEATURE_COLS].to_numpy()

    np.savez_compressed(
        out_dir / f"US_Industrial_Landscape_{tag}_tensor.npz",
        data=tensor,
        dates=np.array([str(pd.Timestamp(d).date()) for d in dates]),
        node_ids=np.array(node_ids),
        node_names=np.array([x[1] for x in NODE_CONFIG]),
        feature_names=np.array(FEATURE_COLS),
    )

    metadata = {
        "tag": tag,
        "shape": list(tensor.shape),
        "axis_order": ["node", "feature", "time"],
        "start_date": str(pd.Timestamp(dates[0]).date()),
        "end_date": str(pd.Timestamp(dates[-1]).date()),
        "missing_values": int(np.isnan(tensor).sum()),
        "sources": SOURCES,
        "node_config": NODE_CONFIG,
        "features": FEATURE_COLS,
    }
    (out_dir / f"US_Industrial_Landscape_{tag}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"[{tag}] shape={tensor.shape}  "
        f"{metadata['start_date']} → {metadata['end_date']}  "
        f"NaN={metadata['missing_values']}"
    )


def main() -> int:
    out_dir = Path.cwd()
    end_year = pd.Timestamp.today().year
    registration_key = None  # Optional: set your BLS registration key here.

    ces_lookup: Dict[Tuple[str, str], str] = {}
    for node_id, _, ces_code, _, _ in NODE_CONFIG:
        for feature, suffix in CES_SUFFIXES.items():
            ces_lookup[(node_id, feature)] = f"CES{ces_code}000000{suffix}"

    jolts_lookup = jolts_series_mapping()

    requested_ids = sorted(
        set(ces_lookup.values())
        | set(jolts_lookup.values())
        | {CPI_SERIES}
    )
    bls = fetch_bls_series(
        requested_ids,
        start_year=START_YEAR,
        end_year=end_year,
        registration_key=registration_key,
    )
    bls_wide = bls.pivot(index="date", columns="series_id", values="value").sort_index()

    bfs = load_bfs()
    output_rows = []

    for node_id, node_name, _, jolts_industry, bfs_codes in NODE_CONFIG:
        bfs_node = (
            bfs[bfs["naics_sector"].isin(bfs_codes)]
            .groupby(["date", "series"], as_index=False)["value"]
            .sum()
            .pivot(index="date", columns="series", values="value")
            .sort_index()
        )

        node = pd.DataFrame(index=bls_wide.index.copy())
        node["employment_level_thousands"] = bls_wide[
            ces_lookup[(node_id, "employment_level_thousands")]
        ]
        node["average_weekly_hours"] = bls_wide[
            ces_lookup[(node_id, "average_weekly_hours")]
        ]
        nominal_ahe = bls_wide[ces_lookup[(node_id, "average_hourly_earnings_nominal")]]
        cpi = bls_wide[CPI_SERIES]
        node["real_average_hourly_earnings_1982_84_dollars"] = nominal_ahe / cpi * 100.0

        for feature in JOLTS_ELEMENTS:
            node[feature] = bls_wide[jolts_lookup[(jolts_industry, feature)]]

        # Left join: BLS dates are kept; BFS-derived columns are NaN where absent.
        node = node.join(bfs_node[["BA_BA", "BA_HBA", "BA_WBA"]], how="left")
        node["business_application_intensity_per_10000_employees"] = (
            node["BA_BA"] / node["employment_level_thousands"] * 10.0
        )
        node["high_propensity_application_share"] = node["BA_HBA"] / node["BA_BA"]
        node["planned_wage_application_share"] = node["BA_WBA"] / node["BA_BA"]
        node = node.drop(columns=["BA_BA", "BA_HBA", "BA_WBA"])
        node["node_id"] = node_id
        node["node_name"] = node_name
        node = node.reset_index().rename(columns={"index": "date"})
        output_rows.append(node)

    panel = pd.concat(output_rows, ignore_index=True)
    panel = panel[panel["date"] >= pd.Timestamp(START_YEAR, 1, 1)].copy()

    # --- raw: all months from START_YEAR, NaN where data is absent ---
    save_outputs(panel, tag="raw", out_dir=out_dir)

    # --- complete: longest contiguous block where every node is fully observed ---
    counts = panel.groupby("date")[FEATURE_COLS].apply(
        lambda x: x.notna().all().all()
    )
    valid_dates = counts[counts].index
    if valid_dates.empty:
        raise RuntimeError("No fully complete common month was found.")

    c_start, c_end = longest_contiguous_run(valid_dates)
    complete = panel[panel["date"].between(c_start, c_end)].copy()
    print(
        f"  Complete window: {c_start.date()} to {c_end.date()} "
        f"({len(complete['date'].unique())} months)"
    )

    if complete[FEATURE_COLS].isna().any().any():
        raise RuntimeError("Missing values remain in the complete panel.")
    if complete.duplicated(["date", "node_id"]).any():
        raise RuntimeError("Duplicate date-node rows remain.")
    if complete.groupby("date")["node_id"].nunique().min() != len(NODE_CONFIG):
        raise RuntimeError("At least one month does not contain all nodes.")

    save_outputs(complete, tag="complete", out_dir=out_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
