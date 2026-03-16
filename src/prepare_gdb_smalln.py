from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def is_spectral_col(col: str) -> bool:
    """True if column name can be parsed as a wavenumber."""
    s = str(col).strip().replace(",", ".")
    try:
        float(s)
        return True
    except Exception:
        return False


def normalize_code(x: object) -> str:
    """Normalize categorical labels (strip, uppercase, fix Cyrillic lookalikes)."""
    s = str(x).strip().upper()
    # Частая проблема: кириллическая С вместо латинской C
    s = s.replace("С", "C")
    # На всякий случай (если вдруг встретятся)
    s = s.replace("Н", "H").replace("О", "O").replace("М", "M").replace("Р", "P")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Path to Data_spectra.xlsx")
    ap.add_argument("--sheet", default="initial", choices=["initial", "data", "all"])
    ap.add_argument("--out", required=True, help="Output parquet path")
    ap.add_argument("--drop-ids", default="", help="Comma-separated sample_id values to drop (optional)")
    args = ap.parse_args()

    xlsx = Path(args.xlsx)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(xlsx, sheet_name=args.sheet)
    df.columns = [str(c).strip() for c in df.columns]

    # Normalize spectral column names (comma -> dot)
    rename_map = {}
    for c in df.columns:
        if is_spectral_col(c):
            rename_map[c] = str(c).strip().replace(",", ".")
    if rename_map:
        df = df.rename(columns=rename_map)

    # Normalize metadata strings
    for c in ["Gender", "Age_factor", "caries_factor", "Parodont", "Anamnes_factor", "Anamnes", "caries", "Name", "ID_new"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    for c in ["Gender", "Age_factor", "caries_factor", "Parodont", "Anamnes_factor"]:
        if c in df.columns:
            df[c] = df[c].map(normalize_code)

    # Sample ID (for group-safe splits)
    if "ID" in df.columns:
        sid = df["ID"].astype(str).str.extract(r"(\d+)")[0]
    elif "Name" in df.columns:
        sid = df["Name"].astype(str).str.extract(r"(\d+)")[0]  # if Name starts with numeric id
    else:
        sid = None

    if sid is None or sid.isna().all():
        df["sample_id"] = [f"s{i+1}" for i in range(len(df))]
    else:
        fallback = pd.Series(range(1, len(df) + 1), index=df.index).astype(str)
        df["sample_id"] = sid.fillna(fallback).astype(str)

    # Useful extra identifiers
    if "ID_new" in df.columns:
        df["sample_code"] = df["ID_new"].astype(str).str.strip()
    elif "Name" in df.columns:
        df["sample_code"] = df["Name"].astype(str).str.strip()
    else:
        df["sample_code"] = df["sample_id"]

    # Pattern code = factor-code without trailing replicate number (e.g., FYCHH1 -> FYCHH)
    df["pattern_code"] = df["sample_code"].astype(str).str.replace(r"\d+$", "", regex=True)

    # Optional drop IDs (for sensitivity analysis, e.g. 17-vs-18)
    if args.drop_ids.strip():
        drop_set = {x.strip() for x in args.drop_ids.split(",") if x.strip()}
        df = df[~df["sample_id"].astype(str).isin(drop_set)].copy()

    # Binary labels (1 = positive/pathology)
    # Parodont: H vs any pathology (G/L/PL/PM/M/etc.)
    df["y_parodont_H_vs_path"] = (df["Parodont"] != "H").astype(int)

    # Anamnes: H vs any pathology (G/C/M/etc.)
    df["y_anamnes_H_vs_path"] = (df["Anamnes_factor"] != "H").astype(int)

    # Caries: healthy H vs any caries-like state (C/F/M/S)
    df["y_caries_H_vs_path"] = (df["caries_factor"] != "H").astype(int)

    # Caries: exact C vs non-C (опционально, если нужен отдельный сценарий)
    df["y_caries_C_vs_nonC"] = (df["caries_factor"] == "C").astype(int)

    # Stress-test: fully healthy across all three axes vs any pathology
    healthy_all = (
        (df["Parodont"] == "H")
        & (df["Anamnes_factor"] == "H")
        & (df["caries_factor"] == "H")
    )
    df["y_healthy_vs_any"] = (~healthy_all).astype(int)

    # Check spectral columns
    spectral_cols = [c for c in df.columns if is_spectral_col(c)]
    if not spectral_cols:
        raise RuntimeError("No spectral columns found (numeric column names expected).")

    # Sort spectral columns numerically for sanity (doesn't change DataFrame order, only print)
    spectral_vals = sorted(float(c) for c in spectral_cols)

    # Save parquet
    df.to_parquet(out, index=False)

    print(f"Saved: {out}")
    print(f"Rows: {len(df)} | Spectral cols: {len(spectral_cols)} | Range: {spectral_vals[0]}..{spectral_vals[-1]}")
    print("sample_id:", df["sample_id"].tolist())
    print("sample_code:", df["sample_code"].tolist())
    print("pattern_code counts:", df["pattern_code"].value_counts().to_dict())

    for ycol in [
        "y_parodont_H_vs_path",
        "y_anamnes_H_vs_path",
        "y_caries_H_vs_path",
        "y_caries_C_vs_nonC",
        "y_healthy_vs_any",
    ]:
        print(ycol, df[ycol].value_counts().to_dict())


if __name__ == "__main__":
    main()