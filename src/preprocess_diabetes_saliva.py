# src/preprocess_diabetes_saliva.py
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


# Repo root: .../nir-ftir
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "diabetes_saliva.parquet"


def _read_table(path: Path) -> pd.DataFrame:
    """Read CSV/XLSX with best-effort defaults."""
    suf = path.suffix.lower()
    if suf in (".csv", ".txt"):
        return pd.read_csv(path)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def _find_file(raw_roots: list[Path], keywords: list[str]) -> Path | None:
    """
    Find a file in raw_roots that matches ALL keywords in its name (case-insensitive),
    searching CSV/XLSX. If multiple matches exist, take the most recently modified.
    """
    candidates: list[Path] = []
    for rr in raw_roots:
        if not rr.exists():
            continue
        for ext in ("*.csv", "*.CSV", "*.xlsx", "*.XLSX", "*.xls", "*.XLS"):
            for p in rr.rglob(ext):
                name = p.name.lower()
                if all(k.lower() in name for k in keywords):
                    candidates.append(p)

    if not candidates:
        return None
    # pick latest modified (usually the right one if duplicates exist)
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def load_single(path: Path, population_label: str) -> pd.DataFrame:
    df = _read_table(path)

    # normalize headers
    df.columns = [str(c).strip().lower() for c in df.columns]

    # required meta
    for col in ("gender", "age"):
        if col not in df.columns:
            raise ValueError(f"В файле {path} нет обязательного столбца '{col}'")

    # align schema
    for col in ("hemoglobin", "glucose"):
        if col not in df.columns:
            df[col] = np.nan

    df["population"] = population_label
    return df


def build_wide(diabetes_path: Path, control_path: Path) -> pd.DataFrame:
    df_diab = load_single(diabetes_path, "DIABETES")
    df_ctrl = load_single(control_path, "CONTROL")

    df_diab["source_file"] = diabetes_path.name
    df_ctrl["source_file"] = control_path.name

    df = pd.concat([df_diab, df_ctrl], ignore_index=True)

    # technical row id
    df.insert(0, "sample_id", [f"S{i:04d}" for i in range(len(df))])

    # IMPORTANT: there is NO patient_id in source -> sample-level split.
    # For pipeline compatibility we provide ID = sample_id.
    df.insert(1, "ID", df["sample_id"].astype(str))

    # target
    df["target"] = (df["population"] == "DIABETES").astype(int)

    # glucose groups (only meaningful for DIABETES; CONTROL gets separate label)
    bins = [45, 100, 150, 200, 250, 300, np.inf]
    labels = [
        "1:[45-100)", "2:[100-150)", "3:[150-200)",
        "4:[200-250)", "5:[250-300)", "6:>=300",
    ]
    df["glucose_group"] = pd.cut(df["glucose"], bins=bins, labels=labels, right=False).astype("string")
    df.loc[df["population"] == "CONTROL", "glucose_group"] = "0:CONTROL"

    meta_cols = {
        "sample_id", "ID", "target", "population",
        "gender", "age", "hemoglobin", "glucose",
        "glucose_group", "source_file",
    }

    # spectral columns = numeric wavenumbers
    spectral_cols = [c for c in df.columns if c not in meta_cols]

    numeric_cols = []
    other_cols = []
    for c in spectral_cols:
        try:
            float(str(c))
            numeric_cols.append(c)
        except Exception:
            other_cols.append(c)

    numeric_cols_sorted = sorted(numeric_cols, key=lambda x: float(str(x)))

    ordered = [
        "sample_id", "ID", "target", "population", "gender", "age",
        "hemoglobin", "glucose", "glucose_group", "source_file",
    ] + other_cols + numeric_cols_sorted

    return df[ordered]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diabetes", type=str, default=None, help="Path to diabetes table (csv/xlsx).")
    ap.add_argument("--control", type=str, default=None, help="Path to control table (csv/xlsx).")
    ap.add_argument("--raw-dir", type=str, default=None, help="Optional raw dir override.")
    args = ap.parse_args()

    # where to search by default
    raw_default_1 = ROOT / "data" / "raw" / "diabetes_saliva"
    raw_default_2 = ROOT / "data" / "raw"
    raw_roots = [Path(args.raw_dir)] if args.raw_dir else [raw_default_1, raw_default_2]

    # resolve paths
    diabetes_path = Path(args.diabetes) if args.diabetes else None
    control_path = Path(args.control) if args.control else None

    if diabetes_path is None:
        diabetes_path = _find_file(raw_roots, ["diabetes"]) or _find_file(raw_roots, ["type", "2"])
    if control_path is None:
        control_path = _find_file(raw_roots, ["control"])

    if diabetes_path is None or not diabetes_path.exists():
        raise FileNotFoundError(
            "Не найден файл DIABETES. Положи его в data/raw или data/raw/diabetes_saliva.\n"
            "Например: data/raw/TYPE 2 DIABETES DATASET.csv"
        )
    if control_path is None or not control_path.exists():
        raise FileNotFoundError(
            "Не найден файл CONTROL. Положи его в data/raw или data/raw/diabetes_saliva.\n"
            "Например: data/raw/CONTROL DATASET.csv"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = build_wide(diabetes_path, control_path)

    print("DIABETES file:", diabetes_path)
    print("CONTROL  file:", control_path)
    print("Final shape:", df.shape)
    print("Target counts:\n", df["target"].value_counts(dropna=False))
    print("ID unique:", df["ID"].nunique(), "rows:", len(df))

    df.to_parquet(PROCESSED_PATH, index=False)
    print("Saved ->", PROCESSED_PATH)


if __name__ == "__main__":
    main()
