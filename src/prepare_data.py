# src/prepare_data.py
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
INTERIM = Path("data/interim")
PROC = Path("data/processed")
for p in (INTERIM, PROC):
    p.mkdir(parents=True, exist_ok=True)


def spectral_columns(cols):
    return [c for c in cols if re.fullmatch(r"\d+(\.\d+)?", str(c).strip())]


def clean_id(s: pd.Series) -> pd.Series:
    # в Source 1 встречаются строки вида "'N1'": убираем крайние апострофы/пробелы
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"^'+|'+$", "", regex=True)
        .str.replace(" ", "", regex=False)
    )


def read_source1(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    df = df.rename(columns=lambda x: str(x).strip())
    if "ID" in df.columns:
        df["ID"] = clean_id(df["ID"])
    specs = spectral_columns(df.columns)

    # метка из Label или N/P
    if "Label" in df.columns:
        y = df["Label"].astype(str).str.strip().map({"Negative": 0, "Positive": 1})
    elif "N/P" in df.columns:
        y = pd.to_numeric(df["N/P"], errors="coerce")
    else:
        y = pd.Series([None] * len(df))
    df["y"] = y.astype("Int64")

    # числовой Ct (на всякий)
    if "Ct" in df.columns:
        df["Ct"] = pd.to_numeric(df["Ct"], errors="coerce")

    df.to_parquet(INTERIM / "source1_raw.parquet", index=False)
    return df, specs


def read_source2(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    df = df.rename(columns=lambda x: str(x).strip())
    if "ID" in df.columns:
        df["ID"] = clean_id(df["ID"])
    specs = spectral_columns(df.columns)
    if not specs:
        raise ValueError("В Source Data 2 не найдены спектральные колонки")

    # метка из префикса ID: P* -> 1, N* -> 0
    if "y" not in df.columns:
        df["y"] = df["ID"].astype(str).str.upper().str[0].map({"P": 1, "N": 0}).astype("Int64")

    df.to_parquet(INTERIM / "source2_raw.parquet", index=False)
    return df, specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src1", default=str(RAW / "Sorce Data 1.xlsx"))
    ap.add_argument("--src2", default=str(RAW / "Source Data 2.xlsx"))
    ap.add_argument(
        "--avg-replicates",
        action="store_true",
        help="Кроме train_repl.parquet создать train_avg.parquet и сделать train.parquet",
    )
    args = ap.parse_args()

    s1 = Path(args.src1)
    s2 = Path(args.src2)
    if not s1.exists():
        raise FileNotFoundError(f"Не найден Source Data 1: {s1}")
    if not s2.exists():
        raise FileNotFoundError(f"Не найден Source Data 2: {s2}")

    df1, specs1 = read_source1(s1)
    df2, specs2 = read_source2(s2)

    # пересечение волновых чисел и общий порядок
    common = sorted(set(specs1).intersection(specs2), key=lambda x: float(x))
    if len(common) < 50:
        print(f"WARNING: мало общих спектральных колонок: {len(common)}")

    # --- TRAIN из Source2 (для классификации)
    train_repl = df2[(["ID"] if "ID" in df2.columns else []) + common + ["y"]].copy()
    # приведение типов
    train_repl[common] = train_repl[common].apply(pd.to_numeric, errors="coerce")

    # среднее по ID (как в статье «Average of 3 replicates»)
    if "ID" in train_repl.columns and args.avg_replicates:
        grp = train_repl.groupby(["ID", "y"], as_index=False)[common].mean()
        train_avg = grp.copy()
        train_avg.to_parquet(PROC / "train_avg.parquet", index=False)
        # для совместимости: train.parquet = усреднённый
        train_avg.to_parquet(PROC / "train.parquet", index=False)
    else:
        # если не усредняем — train.parquet = реплики
        train_repl.to_parquet(PROC / "train.parquet", index=False)

    train_repl.to_parquet(PROC / "train_repl.parquet", index=False)

    # --- EXTERNAL из Source1 (для PCA/нагрузок/EDA)
    external = df1[
        (["ID"] if "ID" in df1.columns else []) + common + (["y"] if "y" in df1.columns else [])
    ].copy()
    external[common] = external[common].apply(pd.to_numeric, errors="coerce")
    external.to_parquet(PROC / "external.parquet", index=False)

    # служебные файлы
    (INTERIM / "columns_source1.txt").write_text("\n".join(map(str, df1.columns)), encoding="utf-8")
    (INTERIM / "columns_source2.txt").write_text("\n".join(map(str, df2.columns)), encoding="utf-8")
    (INTERIM / "common_wavenumbers.txt").write_text("\n".join(map(str, common)), encoding="utf-8")

    # краткая сводка
    print(f"Train_repl shape: {train_repl.shape}")
    if (PROC / "train_avg.parquet").exists():
        ta = pd.read_parquet(PROC / "train_avg.parquet")
        print(f"Train_avg shape:  {ta.shape}")
        print("Label counts (avg):")
        print(ta["y"].value_counts(dropna=False))
    print("Label counts (repl):")
    print(train_repl["y"].value_counts(dropna=False))
    print(f"External shape:    {external.shape}")


if __name__ == "__main__":
    main()
