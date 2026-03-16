from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# scripts/aggregate_reports.py -> корень репозитория = parents[1]
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "exp"


def _f(x: Any) -> float:
    """Safe float conversion (keeps NaN on errors)."""
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _safe_get(d: dict, key: str, default=None):
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def _infer_run_root(json_path: Path) -> str:
    """
    Ищем папку запуска вида run_YYYYMMDD_HHMMSS в пути.
    Если не нашли — возвращаем ближайшую разумную папку.
    """
    for p in json_path.parents:
        name = p.name
        if name.startswith("run_"):
            return name
    # fallback
    if len(json_path.parents) > 1:
        return json_path.parents[1].name
    return ""


def infer_scenario_from_path_and_cfg(json_path: Path, obj: dict) -> str:
    """
    Удобное имя сценария:
    - сначала берём имя папки (A1_covid_baseline_d0 / B2_diab_strong_aug / ...)
    - если папка неинформативна, пытаемся восстановить по config/aug.
    """
    parent_name = json_path.parent.name
    if parent_name and parent_name not in {"exp", "reports"} and not parent_name.startswith("run_"):
        return parent_name

    cfg = obj.get("config", {}) or {}
    aug = obj.get("selected_aug", {}) or {}

    dataset = str(obj.get("dataset", "") or cfg.get("dataset", "")).lower()
    sg_deriv = _safe_get(cfg, "sg_deriv", None)

    # признаки аугментации
    mixwithin = float(_safe_get(aug, "mixwithin", 0.0) or 0.0)
    mixup = float(_safe_get(aug, "mixup", 0.0) or 0.0)
    noise_med = float(_safe_get(aug, "noise_med", 0.0) or 0.0)
    shift = float(_safe_get(aug, "shift", 0.0) or 0.0)
    frda = bool(
        _safe_get(cfg, "frda_lite", False)
        or _safe_get((cfg.get("frda", {}) or {}), "enabled", False)
    )

    if "covid" in dataset:
        deriv_tag = f"d{sg_deriv}" if sg_deriv is not None else "d?"
        if frda:
            return f"covid_classic_frda_{deriv_tag}"
        if mixup > 0 or noise_med > 0 or shift > 0:
            return f"covid_classic_aug_{deriv_tag}"
        return f"covid_baseline_{deriv_tag}"

    if "diabetes" in dataset or "diab" in dataset:
        if mixwithin > 0:
            return "diab_strong_aug"
        return "diab_baseline"

    return "unknown"


def flatten_report(json_path: Path) -> list[dict]:
    try:
        obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    dataset = obj.get("dataset", "")
    protocol = obj.get("protocol", "")
    seed = obj.get("seed", None)
    summary = obj.get("summary", {}) or {}
    config = obj.get("config", {}) or {}
    selected_aug = obj.get("selected_aug", {}) or {}

    scenario = infer_scenario_from_path_and_cfg(json_path, obj)
    run_root = _infer_run_root(json_path)

    metric_cols = ["auc", "pr_auc", "f1", "recall", "prec", "spec", "acc", "brier", "ece"]
    rows: list[dict] = []

    base_common = {
        "json_path": str(json_path),
        "run_root": run_root,               # напр. run_20260225_072002
        "scenario": scenario,               # напр. A1_covid_baseline_d0
        "dataset": dataset,
        "protocol": protocol,
        "seed": seed,
        # полезные поля для фильтрации/анализа
        "group_col": config.get("group_col"),
        "threshold_by": config.get("threshold_by"),
        "recall_target": config.get("recall_target"),
        "min_spec": config.get("min_spec"),
        "sg_deriv": config.get("sg_deriv"),
        "meta_stratify": config.get("meta_stratify", config.get("stratify_meta")),
        "age_bins": config.get("age_bins", config.get("age_bin")),
        # аугментации (если есть)
        "aug_search": selected_aug.get("search_aug"),
        "aug_p_apply": _f(selected_aug.get("p_apply")),
        "aug_noise_std": _f(selected_aug.get("noise_std")),
        "aug_noise_med": _f(selected_aug.get("noise_med")),
        "aug_shift": _f(selected_aug.get("shift")),
        "aug_scale": _f(selected_aug.get("scale")),
        "aug_tilt": _f(selected_aug.get("tilt")),
        "aug_offset": _f(selected_aug.get("offset")),
        "aug_mixup": _f(selected_aug.get("mixup")),
        "aug_mixwithin": _f(selected_aug.get("mixwithin")),
        "aug_repeats": _f(selected_aug.get("aug_repeats")),
    }

    # ---------- MCDCV: summary = {"mean": {...}, "std": {...}} ----------
    if isinstance(summary, dict) and isinstance(summary.get("mean"), dict):
        mean_d = summary.get("mean", {}) or {}
        std_d = summary.get("std", {}) or {}

        for model, m in mean_d.items():
            if not isinstance(m, dict):
                continue

            s = std_d.get(model, {})
            if not isinstance(s, dict):
                s = {}

            row = {
                **base_common,
                "model": model,
                "metric_source": "mcdcv_mean",
                "thr": float("nan"),  # у MCDCV summary единого порога обычно нет
            }

            for k in metric_cols:
                row[k] = _f(m.get(k))
                row[f"{k}_std"] = _f(s.get(k))

            rows.append(row)

        return rows

    # ---------- Holdout: обычно report["results"][model]["test"] ----------
    if isinstance(obj.get("results"), dict):
        for model, model_block in obj["results"].items():
            if not isinstance(model_block, dict):
                continue

            t = model_block.get("test", model_block)
            if not isinstance(t, dict):
                continue

            # минимальная проверка, что это блок метрик
            if ("f1" not in t) and ("auc" not in t) and ("recall" not in t):
                continue

            row = {
                **base_common,
                "model": model,
                "metric_source": "holdout_test",
                "thr": _f(t.get("thr")),
            }

            for k in metric_cols:
                row[k] = _f(t.get(k))
                row[f"{k}_std"] = float("nan")  # один holdout JSON -> std нет

            rows.append(row)

        return rows

    # ---------- fallback: старая структура в summary ----------
    if isinstance(summary, dict):
        candidate_models = summary.get("test", summary) if isinstance(summary.get("test"), dict) else summary

        for model, t in candidate_models.items():
            if not isinstance(t, dict):
                continue
            if ("f1" not in t) and ("auc" not in t) and ("recall" not in t):
                continue

            row = {
                **base_common,
                "model": model,
                "metric_source": "holdout_test",
                "thr": _f(t.get("thr")),
            }

            for k in metric_cols:
                row[k] = _f(t.get(k))
                row[f"{k}_std"] = float("nan")

            rows.append(row)

    return rows


def main() -> None:
    if not REPORTS.exists():
        raise SystemExit(f"Reports dir not found: {REPORTS}")

    # Можно ограничить только одним запуском:
    # ONLY_SUBDIR=run_20260225_072002 python scripts/aggregate_reports.py
    only_subdir = os.environ.get("ONLY_SUBDIR", "").strip()

    if only_subdir:
        target = REPORTS / only_subdir
        if not target.exists():
            raise SystemExit(f"ONLY_SUBDIR not found: {target}")
        search_roots = [target]
    else:
        search_roots = [REPORTS]

    # Берём только seed-отчёты, чтобы не тянуть посторонние JSON
    json_files: list[Path] = []
    for root in search_roots:
        json_files.extend([p for p in sorted(root.rglob("*.json")) if "_seed" in p.name])

    if not json_files:
        roots_str = ", ".join(str(r) for r in search_roots)
        raise SystemExit(f"No *_seed*.json found under: {roots_str}")

    all_rows: list[dict] = []
    for p in json_files:
        all_rows.extend(flatten_report(p))

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No rows found after parsing JSON reports.")
        return

    metric_cols = ["auc", "pr_auc", "f1", "recall", "prec", "spec", "acc", "brier", "ece"]

    ordered = [
        "json_path",
        "run_root",
        "scenario",
        "dataset",
        "protocol",
        "seed",
        "model",
        "metric_source",
        *metric_cols,
        *[f"{m}_std" for m in metric_cols],
        "thr",
        "threshold_by",
        "recall_target",
        "min_spec",
        "group_col",
        "sg_deriv",
        "meta_stratify",
        "age_bins",
        "aug_search",
        "aug_p_apply",
        "aug_noise_std",
        "aug_noise_med",
        "aug_shift",
        "aug_scale",
        "aug_tilt",
        "aug_offset",
        "aug_mixup",
        "aug_mixwithin",
        "aug_repeats",
    ]

    for c in ordered:
        if c not in df.columns:
            df[c] = np.nan

    df = df[ordered + [c for c in df.columns if c not in ordered]]

    # ---- output paths ----
    out_runs = ROOT / "summary_runs_from_json.csv"
    out_reports_dir = ROOT / "reports"
    out_reports_dir.mkdir(exist_ok=True, parents=True)

    out_grouped = out_reports_dir / "summary_grouped.csv"
    out_hold_best = out_reports_dir / "summary_grouped_holdout.csv"
    out_hold_mean = out_reports_dir / "summary_grouped_holdout_mean.csv"

    # -------- Flat rows --------
    df.to_csv(out_runs, index=False, encoding="utf-8-sig")

    # -------- Grouped MCDCV (mean/std/count across seed files) --------
    mcdcv = df[df["metric_source"] == "mcdcv_mean"].copy()
    if not mcdcv.empty:
        grp_cols = ["run_root", "scenario", "dataset", "protocol", "model"]
        agg_dict = {m: ["mean", "std", "count"] for m in metric_cols}

        g = mcdcv.groupby(grp_cols, dropna=False).agg(agg_dict)
        g.columns = [f"{a}_{b}" for a, b in g.columns]
        g = g.reset_index()
        g.to_csv(out_grouped, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(out_grouped, index=False, encoding="utf-8-sig")

    # -------- Holdout grouped --------
    hold = df[df["metric_source"] == "holdout_test"].copy()
    if not hold.empty:
        grp_cols = ["run_root", "scenario", "dataset", "protocol", "model"]

        # best-by-F1 (по каждому run/scenario/model)
        hold_best = (
            hold.sort_values(
                ["run_root", "scenario", "dataset", "protocol", "model", "f1"],
                ascending=[True, True, True, True, True, False],
            )
            .groupby(grp_cols, dropna=False, as_index=False)
            .first()
        )
        hold_best.to_csv(out_hold_best, index=False, encoding="utf-8-sig")

        # mean/std/count по seed для holdout
        agg_dict = {m: ["mean", "std", "count"] for m in metric_cols}
        h = hold.groupby(grp_cols, dropna=False).agg(agg_dict)
        h.columns = [f"{a}_{b}" for a, b in h.columns]
        h = h.reset_index()
        h.to_csv(out_hold_mean, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(out_hold_best, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(out_hold_mean, index=False, encoding="utf-8-sig")

    print(f"[OK] Parsed JSON files: {len(json_files)}")
    print(f"[OK] Saved: {out_runs} (rows={len(df)})")
    print(f"[OK] Saved: {out_grouped}")
    print(f"[OK] Saved: {out_hold_best}")
    print(f"[OK] Saved: {out_hold_mean}")


if __name__ == "__main__":
    main()