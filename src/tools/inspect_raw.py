from pathlib import Path

import pandas as pd

RAW = Path("data/raw")


def head_preview(path: Path, rows: int = 5, cols: int = 10) -> None:
    print(f"\n=== PREVIEW: {path.name} ===\n")
    # читаем без заголовка и с заголовком — чтобы увидеть первую строку и типы
    df_nohdr = pd.read_excel(path, sheet_name=0, header=None, engine="openpyxl")
    df_hdr = pd.read_excel(path, sheet_name=0, header=0, engine="openpyxl")
    print("[header=None] top-left:")
    print(df_nohdr.iloc[:rows, :cols])
    print("\n[header=0] columns:")
    print(list(df_hdr.columns)[:20])
    print("dtypes (первые 12):", list(df_hdr.dtypes[:12]))


def main():
    files = sorted(p for p in RAW.glob("*.xlsx") if not p.name.startswith("~$"))
    if not files:
        print("Файлы *.xlsx не найдены в data/raw")
        return
    for p in files:
        head_preview(p)


if __name__ == "__main__":
    main()
