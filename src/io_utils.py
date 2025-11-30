# src/io_utils.py
from __future__ import annotations

from pathlib import Path

import pandas as pd


def excel_overview(path: Path):
    """Вернёт список листов в книге Excel."""
    xls = pd.ExcelFile(path)
    return xls.sheet_names


def read_ftir_sheet(
    path: Path,
    sheet: int | str = 0,
    meta_cols: int = 3,
    meta_names: tuple[str, ...] = ("sample_id", "patient_id", "label"),
):
    """
    Ожидаемый формат:
      - строка 0: волновые числа (начиная с колонки meta_cols)
      - строки 1..: спектры (числа)
      - первые meta_cols колонок: метаданные.
    Подгони meta_cols/meta_names под свой файл, если нужно.
    """
    df = pd.read_excel(path, sheet_name=sheet, header=None)
    wn = df.iloc[0, meta_cols:].astype(float).to_numpy()
    meta = df.iloc[1:, :meta_cols].copy()
    meta.columns = list(meta_names)[:meta_cols]
    X = df.iloc[1:, meta_cols:].astype(float).to_numpy()
    return wn, X, meta
