# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "reports" / "exp"
OUT_CSV = EXP_DIR / "summary.csv"

wanted = {
    "config.json": "config",
    "cv_metrics.csv": "cv",
    "test_metrics.csv": "test",
}

rows = []
for exp in sorted([p for p in EXP_DIR.iterdir() if p.is_dir()]):
    blobs = {}
    for fname, key in wanted.items():
        f = exp / fname
        if not f.exists():
            continue
        if key == "config":
            blobs[key] = json.loads(f.read_text(encoding="utf-8"))
        else:
            blobs[key] = pd.read_csv(f)
    if "config" not in blobs or "cv" not in blobs or "test" not in blobs:
        continue

    cfg = blobs["config"]
    # средние по фолдам для каждой модели
    cv_mean = blobs["cv"].groupby("model").mean(numeric_only=True).reset_index()
    for _, row in cv_mean.iterrows():
        model = row["model"]
        test = blobs["test"][blobs["test"]["model"] == model]
        test_row = test.iloc[0].to_dict() if len(test) else {}
        rows.append(
            {
                "ts": exp.name,
                "model": model,
                "norm": cfg.get("norm"),
                "dataset": cfg.get("dataset"),
                "crop_min": cfg.get("crop_min"),
                "crop_max": cfg.get("crop_max"),
                "noise": cfg.get("noise"),
                "shift": cfg.get("shift_cm"),
                "mixup": cfg.get("mixup"),
                "cv_roc_auc_mean": row.get("roc_auc"),
                "cv_pr_auc_mean": row.get("pr_auc"),
                "test_roc_auc": test_row.get("roc_auc"),
                "test_pr_auc": test_row.get("pr_auc"),
                "test_recall_pos": test_row.get("recall_pos"),
                "test_f1_pos": test_row.get("f1_pos"),
            }
        )

df = pd.DataFrame(rows)
df = df.sort_values(["dataset", "ts", "model"])
EXP_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV} (n={len(df)})")
