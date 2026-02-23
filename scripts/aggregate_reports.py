# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports" / "exp"   # у тебя там папки run_YYYYMMDD_HHMMSS/...

def safe_read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def parse_from_filename(name: str) -> dict:
    # пример: run_diabetes_saliva_cv_holdout_seed2_d1_f1_plus_270f21834d.json
    out = {}
    m = re.search(r"(covid_saliva|diabetes_saliva)", name)
    if m: out["dataset"] = m.group(1)
    m = re.search(r"(cv_holdout|mcdcv|mcdcv_plsda)", name)
    if m: out["protocol"] = m.group(1)
    m = re.search(r"seed(-?\d+)", name)
    if m: out["seed"] = int(m.group(1))
    m = re.search(r"_d(\d+)_", name)
    if m: out["sg_deriv"] = int(m.group(1))
    m = re.search(r"_(none|recall|recall_plus|f1_plus)_", name)
    if m: out["threshold_by"] = m.group(1)
    return out

def infer_scenario(cfg: dict) -> str:
    # твоя структура: cfg["aug"] и cfg["frda"]["enabled"]
    aug = (cfg.get("aug") or {})
    frda = (cfg.get("frda") or {})
    frda_on = bool(frda.get("enabled", False))

    # baseline если все нули
    is_no_aug = (
        float(aug.get("noise_std", 0.0)) == 0.0 and
        float(aug.get("noise_med", 0.0)) == 0.0 and
        float(aug.get("shift", 0.0)) == 0.0 and
        float(aug.get("scale", 0.0)) == 0.0 and
        float(aug.get("tilt", 0.0)) == 0.0 and
        float(aug.get("offset", 0.0)) == 0.0 and
        float(aug.get("mixup", 0.0)) == 0.0 and
        float(aug.get("mixwithin", 0.0)) == 0.0 and
        int(aug.get("aug_repeats", 1)) <= 1
    )
    if frda_on:
        return "frda_lite"
    if is_no_aug:
        return "baseline"
    return "aug"

def flatten_report(json_path: Path) -> list[dict]:
    rep = safe_read_json(json_path)
    if not rep:
        return []

    cfg = rep.get("config", {}) or {}
    detected = rep.get("detected", {}) or {}
    scenario = infer_scenario(cfg)

    info = {
        "file": str(json_path),
        "run_dir": str(json_path.parent),
        "dataset": rep.get("config", {}).get("dataset", parse_from_filename(json_path.name).get("dataset", "unknown")),
        "protocol": rep.get("protocol", parse_from_filename(json_path.name).get("protocol", "unknown")),
        "seed": int(rep.get("config", {}).get("seed", parse_from_filename(json_path.name).get("seed", -1))),
        "scenario": scenario,
        "threshold_by": rep.get("config", {}).get("threshold_by", parse_from_filename(json_path.name).get("threshold_by", "")),
        "sg_deriv": int(rep.get("config", {}).get("sg_deriv", parse_from_filename(json_path.name).get("sg_deriv", -1))),
        "n_samples": detected.get("n_samples", np.nan),
        "n_features": detected.get("n_features", np.nan),
        "group_col": detected.get("group_col", None),
        "n_groups": detected.get("n_groups", np.nan),
        "max_reps_per_group": detected.get("max_reps_per_group", np.nan),
    }

    rows = []

    # cv_holdout формат: rep["results"][model]["test"]
    if "results" in rep:
        results = rep["results"] or {}
        for model, r in results.items():
            t = (r.get("test") or {})
            rows.append({
                **info,
                "model": model,
                "metric_source": "holdout_test",
                "thr": float(t.get("thr", t.get("threshold", np.nan))),
                "f1": float(t.get("f1", np.nan)),
                "auc": float(t.get("auc", np.nan)),
                "acc": float(t.get("acc", np.nan)),
                "recall": float(t.get("recall", np.nan)),
                "prec": float(t.get("prec", np.nan)),
                "spec": float(t.get("spec", np.nan)),
                "tp": float(t.get("tp", np.nan)),
                "fp": float(t.get("fp", np.nan)),
                "tn": float(t.get("tn", np.nan)),
                "fn": float(t.get("fn", np.nan)),
            })
        return rows

    # mcdcv формат: rep["summary"]["mean"][model], rep["summary"]["std"][model]
    if "summary" in rep:
        mean = (rep["summary"].get("mean") or {})
        std = (rep["summary"].get("std") or {})
        for model, m in mean.items():
            s = std.get(model, {})
            rows.append({
                **info,
                "model": model,
                "metric_source": "mcdcv_mean",
                "f1": float(m.get("f1", np.nan)),
                "auc": float(m.get("auc", np.nan)),
                "acc": float(m.get("acc", np.nan)),
                "recall": float(m.get("recall", np.nan)),
                "prec": float(m.get("prec", np.nan)),
                "spec": float(m.get("spec", np.nan)),
                "f1_std": float(s.get("f1", np.nan)),
                "auc_std": float(s.get("auc", np.nan)),
                "acc_std": float(s.get("acc", np.nan)),
                "recall_std": float(s.get("recall", np.nan)),
                "prec_std": float(s.get("prec", np.nan)),
                "spec_std": float(s.get("spec", np.nan)),
            })
        return rows

    return []

def main():
    if not EXP.exists():
        raise SystemExit(f"Not found: {EXP}")

    json_files = list(EXP.rglob("*.json"))
    json_files = [p for p in json_files if p.name.startswith("run_") and ("_seed" in p.name)]
    if not json_files:
        raise SystemExit("No run_*.json found under reports/exp")

    rows = []
    for p in json_files:
        rows.extend(flatten_report(p))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Parsed 0 rows (unexpected).")

    out_runs = ROOT / "reports" / "summary_runs_from_json.csv"
    out_runs.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_runs, index=False)

    # группировка: mean±std по seed (для holdout) / уже mean для mcdcv
    # оставим отдельно holdout
    hold = df[df["metric_source"] == "holdout_test"].copy()
    if not hold.empty:
        grp = (
            hold.groupby(["dataset","protocol","scenario","model"], dropna=False)
                .agg(
                    n=("f1","count"),
                    f1_mean=("f1","mean"), f1_std=("f1","std"),
                    auc_mean=("auc","mean"), auc_std=("auc","std"),
                    spec_mean=("spec","mean"), spec_std=("spec","std"),
                    recall_mean=("recall","mean"), recall_std=("recall","std"),
                ).reset_index()
        )
        out_grp = ROOT / "reports" / "summary_grouped_holdout.csv"
        grp.to_csv(out_grp, index=False)
        print("[OK] grouped holdout:", out_grp)

    print("[OK] runs table:", out_runs)
    print(df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
