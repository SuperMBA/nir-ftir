# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports" / "exp"

# фильтр по korea parquet (подстрой при необходимости)
KOREA_KEY = "korea_main_final_bestrep_train"


# -----------------------------
# helpers
# -----------------------------
def safe_read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def as_float(x: Any, default: float = np.nan) -> float:
    try:
        if x is None or x == "":
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def as_int(x: Any, default: int = -1) -> int:
    try:
        if x is None or x == "":
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def get_cfg(cfg: dict, *keys, default=None):
    """Безопасно достаёт значение из вложенного dict."""
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def approx(x: float, y: float, tol: float = 1e-6) -> bool:
    if np.isnan(x) or np.isnan(y):
        return False
    return abs(x - y) <= tol


# -----------------------------
# korea detection / parsing
# -----------------------------
def is_korea(rep: dict, json_path: Path) -> bool:
    # 1) тег запуска
    run_tag = str(rep.get("run_tag", "") or rep.get("tag", "") or "").lower()
    if run_tag == "korea":
        return True

    cfg = rep.get("config", {}) or {}

    # 2) путь к parquet
    dp = str(cfg.get("data_path", "") or cfg.get("dataset_path", "") or "").lower()
    if KOREA_KEY in dp:
        return True

    # 3) запасной фильтр по имени файла / директории
    jn = json_path.name.lower()
    pn = str(json_path.parent).lower()
    if "korea" in jn or "korea" in pn:
        return True

    return False


def parse_from_filename(name: str) -> dict:
    out: dict[str, Any] = {}

    m = re.search(r"(cv_holdout|mcdcv_plsda|mcdcv)", name)
    if m:
        out["protocol"] = m.group(1)

    m = re.search(r"seed(-?\d+)", name)
    if m:
        out["seed"] = int(m.group(1))

    m = re.search(r"_d(\d+)_", name)
    if m:
        out["sg_deriv"] = int(m.group(1))

    m = re.search(r"_(none|recall|recall_plus|f1_plus)_", name)
    if m:
        out["threshold_by"] = m.group(1)

    return out


def infer_scenario(cfg: dict) -> str:
    """
    Различает:
      - baseline
      - classic_aug
      - strong_aug
      - aug_other
      + suffix +frda, если FRDA-lite включён
    """
    aug = (cfg.get("aug") or {})
    frda = (cfg.get("frda") or {})

    # FRDA может лежать в разных местах в зависимости от версии кода
    frda_on = bool(frda.get("enabled", False)) or bool(cfg.get("frda_lite", False))

    noise_std = as_float(aug.get("noise_std", 0.0), 0.0)
    noise_med = as_float(aug.get("noise_med", 0.0), 0.0)
    shift = as_float(aug.get("shift", 0.0), 0.0)
    scale = as_float(aug.get("scale", 0.0), 0.0)
    tilt = as_float(aug.get("tilt", 0.0), 0.0)
    offset = as_float(aug.get("offset", 0.0), 0.0)
    mixup = as_float(aug.get("mixup", 0.0), 0.0)
    mixwithin = as_float(aug.get("mixwithin", 0.0), 0.0)
    aug_repeats = as_int(aug.get("aug_repeats", 1), 1)
    p_apply = as_float(aug.get("p_apply", 0.5), 0.5)

    eps = 1e-12
    is_no_aug = (
        abs(noise_std) < eps and
        abs(noise_med) < eps and
        abs(shift) < eps and
        abs(scale) < eps and
        abs(tilt) < eps and
        abs(offset) < eps and
        abs(mixup) < eps and
        abs(mixwithin) < eps and
        aug_repeats <= 1
    )

    if is_no_aug:
        base = "baseline"
    else:
        # strong preset (из нового скрипта)
        # mixwithin=0.4, scale=0.02, tilt=0.02, offset=0.01, aug_repeats=3, p_apply=0.7
        is_strong = (
            approx(mixwithin, 0.40) and
            approx(scale, 0.02) and
            approx(tilt, 0.02) and
            approx(offset, 0.01) and
            aug_repeats == 3 and
            approx(p_apply, 0.70)
        )

        # classic preset (из нового скрипта)
        # noise_med=0.010, shift=1.0, mixup=0.3, mixwithin=0.2, aug_repeats=1, p_apply=0.5
        is_classic = (
            approx(noise_med, 0.010) and
            approx(shift, 1.0) and
            approx(mixup, 0.30) and
            approx(mixwithin, 0.20) and
            aug_repeats == 1 and
            approx(p_apply, 0.50)
        )

        if is_strong:
            base = "strong_aug"
        elif is_classic:
            base = "classic_aug"
        else:
            base = "aug_other"

    if frda_on:
        return f"{base}+frda"
    return base


def extract_preproc(cfg: dict, fn: dict) -> dict:
    """Пытается достать параметры препроцессинга из разных схем конфигурации."""
    sg_deriv = cfg.get("sg_deriv") if "sg_deriv" in cfg else get_cfg(cfg, "preproc", "sg_deriv", default=None)
    sg_window = cfg.get("sg_window") if "sg_window" in cfg else get_cfg(cfg, "preproc", "sg_window", default=None)
    sg_poly = cfg.get("sg_poly") if "sg_poly" in cfg else get_cfg(cfg, "preproc", "sg_poly", default=None)
    crop_min = cfg.get("crop_min") if "crop_min" in cfg else get_cfg(cfg, "preproc", "crop_min", default=None)
    crop_max = cfg.get("crop_max") if "crop_max" in cfg else get_cfg(cfg, "preproc", "crop_max", default=None)
    norm = cfg.get("norm") if "norm" in cfg else get_cfg(cfg, "preproc", "norm", default=None)
    xscale = cfg.get("xscale") if "xscale" in cfg else get_cfg(cfg, "preproc", "xscale", default=None)

    return {
        "sg_deriv": as_int(sg_deriv, fn.get("sg_deriv", -1)),
        "sg_window": as_int(sg_window, -1),
        "sg_poly": as_int(sg_poly, -1),
        "crop_min": as_float(crop_min, np.nan),
        "crop_max": as_float(crop_max, np.nan),
        "norm": norm or "",
        "xscale": xscale or "",
    }


def flatten_report(json_path: Path) -> list[dict]:
    rep = safe_read_json(json_path)
    if not rep:
        return []
    if not is_korea(rep, json_path):
        return []

    cfg = rep.get("config", {}) or {}
    detected = rep.get("detected", {}) or {}
    fn = parse_from_filename(json_path.name)
    scenario = infer_scenario(cfg)
    prep = extract_preproc(cfg, fn)

    # поля из config (с fallback на filename)
    protocol = str(rep.get("protocol") or cfg.get("protocol") or fn.get("protocol") or "unknown")
    seed = as_int(cfg.get("seed", fn.get("seed", -1)), -1)
    threshold_by = (
        cfg.get("threshold_by")
        or get_cfg(cfg, "threshold", "by", default=None)
        or fn.get("threshold_by")
        or ""
    )

    # aug/frda для отладки/анализа
    aug = (cfg.get("aug") or {})
    frda = (cfg.get("frda") or {})

    # mtime файла для дедупликации (берём последний)
    try:
        file_mtime = json_path.stat().st_mtime
    except Exception:
        file_mtime = np.nan

    info = {
        "file": str(json_path),
        "file_name": json_path.name,
        "file_mtime": file_mtime,
        "run_dir": str(json_path.parent),
        "run_stage": json_path.parent.name,   # K1_..., K2_...
        "dataset": "korea_bestrep",
        "protocol": protocol,
        "seed": seed,
        "scenario": scenario,
        "threshold_by": str(threshold_by),
        "run_tag": str(rep.get("run_tag", "") or rep.get("tag", "") or ""),

        "data_path": str(cfg.get("data_path", "") or cfg.get("dataset_path", "") or ""),
        "group_col": detected.get("group_col", cfg.get("group_col", None)),
        "n_groups": as_float(detected.get("n_groups", np.nan)),
        "max_reps_per_group": as_float(detected.get("max_reps_per_group", np.nan)),
        "n_samples": as_float(detected.get("n_samples", np.nan)),
        "n_features": as_float(detected.get("n_features", np.nan)),

        # protocol params
        "mc_iter": as_int(cfg.get("mc_iter", get_cfg(cfg, "protocol_params", "mc_iter", default=-1)), -1),
        "inner_splits": as_int(cfg.get("inner_splits", get_cfg(cfg, "protocol_params", "inner_splits", default=-1)), -1),
        "val_size": as_float(cfg.get("val_size", get_cfg(cfg, "protocol_params", "val_size", default=np.nan)), np.nan),

        # preproc
        **prep,

        # aug params
        "aug_noise_std": as_float(aug.get("noise_std", 0.0), 0.0),
        "aug_noise_med": as_float(aug.get("noise_med", 0.0), 0.0),
        "aug_shift": as_float(aug.get("shift", 0.0), 0.0),
        "aug_scale": as_float(aug.get("scale", 0.0), 0.0),
        "aug_tilt": as_float(aug.get("tilt", 0.0), 0.0),
        "aug_offset": as_float(aug.get("offset", 0.0), 0.0),
        "aug_mixup": as_float(aug.get("mixup", 0.0), 0.0),
        "aug_mixwithin": as_float(aug.get("mixwithin", 0.0), 0.0),
        "aug_repeats": as_int(aug.get("aug_repeats", 1), 1),
        "aug_p_apply": as_float(aug.get("p_apply", 0.5), 0.5),

        # frda params
        "frda_enabled": bool(frda.get("enabled", False) or cfg.get("frda_lite", False)),
        "frda_k": as_int(frda.get("k", cfg.get("frda_k", -1)), -1),
        "frda_width": as_int(frda.get("width", cfg.get("frda_width", -1)), -1),
        "frda_local_scale": as_float(frda.get("local_scale", cfg.get("frda_local_scale", np.nan)), np.nan),
    }

    rows: list[dict] = []

    # ---- holdout format ----
    if "results" in rep and isinstance(rep["results"], dict):
        results = rep["results"] or {}
        for model, r in results.items():
            if not isinstance(r, dict):
                continue
            t = (r.get("test") or {})
            if not isinstance(t, dict):
                t = {}

            rows.append({
                **info,
                "model": str(model),
                "metric_source": "holdout_test",
                "thr": as_float(t.get("thr", t.get("threshold", np.nan))),
                "f1": as_float(t.get("f1", np.nan)),
                "auc": as_float(t.get("auc", np.nan)),
                "acc": as_float(t.get("acc", np.nan)),
                "recall": as_float(t.get("recall", np.nan)),
                "prec": as_float(t.get("prec", np.nan)),
                "spec": as_float(t.get("spec", np.nan)),
                "f1_std_inrun": np.nan,
                "auc_std_inrun": np.nan,
                "spec_std_inrun": np.nan,
                "recall_std_inrun": np.nan,
            })

        if rows:
            return rows

    # ---- mcdcv format ----
    if "summary" in rep and isinstance(rep["summary"], dict):
        sm = rep["summary"] or {}
        mean = (sm.get("mean") or {})
        std = (sm.get("std") or {})

        if isinstance(mean, dict):
            for model, m in mean.items():
                if not isinstance(m, dict):
                    continue
                s = std.get(model, {}) if isinstance(std, dict) else {}
                if not isinstance(s, dict):
                    s = {}

                rows.append({
                    **info,
                    "model": str(model),
                    "metric_source": "mcdcv_mean",
                    "thr": np.nan,
                    "f1": as_float(m.get("f1", np.nan)),
                    "auc": as_float(m.get("auc", np.nan)),
                    "acc": as_float(m.get("acc", np.nan)),
                    "recall": as_float(m.get("recall", np.nan)),
                    "prec": as_float(m.get("prec", np.nan)),
                    "spec": as_float(m.get("spec", np.nan)),
                    "f1_std_inrun": as_float(s.get("f1", np.nan)),
                    "auc_std_inrun": as_float(s.get("auc", np.nan)),
                    "spec_std_inrun": as_float(s.get("spec", np.nan)),
                    "recall_std_inrun": as_float(s.get("recall", np.nan)),
                })

        if rows:
            return rows

    return []


# -----------------------------
# de-duplication
# -----------------------------
def deduplicate_runs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Убирает дубликаты прогонов и оставляет последний файл по времени (file_mtime).
    Ключ дедупликации: один прогон = (metric_source, run_stage, protocol, seed, scenario, sg_deriv, threshold_by, model)
    """
    if df.empty:
        return df.copy(), df.iloc[0:0].copy()

    dedup_keys = [
        "metric_source",
        "run_stage",
        "protocol",
        "seed",
        "scenario",
        "sg_deriv",
        "threshold_by",
        "model",
    ]

    work = df.copy()
    # stable sort: сначала по ключам, затем по времени файла
    work = work.sort_values(dedup_keys + ["file_mtime", "file"], ascending=True, kind="mergesort")

    dup_mask = work.duplicated(subset=dedup_keys, keep="last")
    dropped = work.loc[dup_mask].copy()
    kept = work.loc[~dup_mask].copy()

    return kept.reset_index(drop=True), dropped.reset_index(drop=True)


# -----------------------------
# aggregation / exports
# -----------------------------
def make_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегация по seed.
    ВАЖНО: metric_source теперь включён в группировку (mcdcv_mean vs holdout_test не смешиваем).
    """
    if df.empty:
        return df.copy()

    grp_cols = ["metric_source", "protocol", "scenario", "model", "sg_deriv", "threshold_by"]

    lb = (
        df.groupby(grp_cols, dropna=False)
        .agg(
            n=("seed", "nunique"),

            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),

            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),

            spec_mean=("spec", "mean"),
            spec_std=("spec", "std"),

            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),

            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),

            prec_mean=("prec", "mean"),
            prec_std=("prec", "std"),

            # Для mcdcv: средний std внутри одного run (по итерациям MC)
            f1_std_inrun_mean=("f1_std_inrun", "mean"),
            auc_std_inrun_mean=("auc_std_inrun", "mean"),
            spec_std_inrun_mean=("spec_std_inrun", "mean"),
            recall_std_inrun_mean=("recall_std_inrun", "mean"),
        )
        .reset_index()
    )

    # красивые колонки "mean ± std"
    for m in ["f1", "auc", "spec", "recall", "acc", "prec"]:
        lb[f"{m}_mean_pm_std"] = (
            lb[f"{m}_mean"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "nan")
            + " ± " +
            lb[f"{m}_std"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "nan")
        )

    # сортировка: сначала mcdcv, потом D1, потом baseline/classic/strong
    metric_source_rank = {"mcdcv_mean": 0, "holdout_test": 1}
    scenario_rank = {
        "baseline": 0,
        "classic_aug": 1,
        "strong_aug": 2,
        "aug_other": 3,
        "baseline+frda": 4,
        "classic_aug+frda": 5,
        "strong_aug+frda": 6,
        "aug_other+frda": 7,
    }

    lb["metric_source_rank"] = lb["metric_source"].map(metric_source_rank).fillna(99).astype(int)
    lb["scenario_rank"] = lb["scenario"].map(scenario_rank).fillna(99).astype(int)

    lb = lb.sort_values(
        ["metric_source_rank", "sg_deriv", "scenario_rank", "f1_mean", "auc_mean", "spec_mean"],
        ascending=[True, False, True, False, False, False]
    ).reset_index(drop=True)

    return lb


def make_headline(lb: pd.DataFrame) -> pd.DataFrame:
    """
    Компактная таблица: best model для каждой комбинации (metric_source, sg_deriv, scenario)
    """
    if lb.empty:
        return lb.copy()

    tmp = lb.sort_values(
        ["metric_source", "sg_deriv", "scenario", "auc_mean", "f1_mean"],
        ascending=[True, False, True, False, False]
    ).copy()

    head = tmp.groupby(["metric_source", "sg_deriv", "scenario"], as_index=False).first()

    head["preproc"] = head["sg_deriv"].map({0: "D0", 1: "D1"}).fillna("unknown")

    cols = [
        "metric_source", "preproc", "sg_deriv", "scenario", "model", "n",
        "auc_mean_pm_std", "f1_mean_pm_std", "recall_mean_pm_std", "spec_mean_pm_std",
        "auc_mean", "f1_mean", "recall_mean", "spec_mean",
        "threshold_by", "protocol"
    ]
    head = head[cols].rename(columns={
        "model": "best_model",
        "n": "n_seeds"
    })

    metric_source_order = pd.Categorical(
        head["metric_source"],
        categories=["mcdcv_mean", "holdout_test"],
        ordered=True
    )
    scenario_order = pd.Categorical(
        head["scenario"],
        categories=[
            "baseline", "classic_aug", "strong_aug", "aug_other",
            "baseline+frda", "classic_aug+frda", "strong_aug+frda", "aug_other+frda"
        ],
        ordered=True
    )

    head = (
        head.assign(_ms=metric_source_order, _sc=scenario_order)
            .sort_values(["_ms", "sg_deriv", "_sc"], ascending=[True, False, True])
            .drop(columns=["_ms", "_sc"])
            .reset_index(drop=True)
    )

    return head


def make_scenario_model_matrix(lb: pd.DataFrame) -> pd.DataFrame:
    """
    Матрица сравнения сценариев по моделям.
    Делаем отдельно по каждому metric_source.
    """
    if lb.empty:
        return lb.copy()

    parts = []
    for ms, sub in lb.groupby("metric_source", dropna=False):
        pivot_f1 = sub.pivot_table(
            index=["sg_deriv", "model"],
            columns="scenario",
            values="f1_mean",
            aggfunc="mean"
        ).reset_index()

        pivot_auc = sub.pivot_table(
            index=["sg_deriv", "model"],
            columns="scenario",
            values="auc_mean",
            aggfunc="mean"
        ).reset_index()

        pivot_f1.columns = [
            f"F1_{c}" if isinstance(c, str) and c not in {"sg_deriv", "model"} else c
            for c in pivot_f1.columns
        ]
        pivot_auc.columns = [
            f"AUC_{c}" if isinstance(c, str) and c not in {"sg_deriv", "model"} else c
            for c in pivot_auc.columns
        ]

        out = pivot_f1.merge(pivot_auc, on=["sg_deriv", "model"], how="outer")
        out.insert(0, "metric_source", ms)

        # Δ-колонки
        if "F1_baseline" in out.columns and "F1_strong_aug" in out.columns:
            out["dF1_strong_vs_base"] = out["F1_strong_aug"] - out["F1_baseline"]
        if "F1_baseline" in out.columns and "F1_classic_aug" in out.columns:
            out["dF1_classic_vs_base"] = out["F1_classic_aug"] - out["F1_baseline"]

        if "AUC_baseline" in out.columns and "AUC_strong_aug" in out.columns:
            out["dAUC_strong_vs_base"] = out["AUC_strong_aug"] - out["AUC_baseline"]
        if "AUC_baseline" in out.columns and "AUC_classic_aug" in out.columns:
            out["dAUC_classic_vs_base"] = out["AUC_classic_aug"] - out["AUC_baseline"]

        parts.append(out)

    mx = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    metric_source_order = {"mcdcv_mean": 0, "holdout_test": 1}
    if not mx.empty:
        mx["metric_source_rank"] = mx["metric_source"].map(metric_source_order).fillna(99)
        mx = mx.sort_values(["metric_source_rank", "sg_deriv", "model"], ascending=[True, False, True]).drop(columns=["metric_source_rank"])

    return mx.reset_index(drop=True)


def main():
    if not EXP.exists():
        raise SystemExit(f"Not found: {EXP}")

    json_files = list(EXP.rglob("*.json"))
    if not json_files:
        raise SystemExit("No *.json found under reports/exp")

    rows = []
    for p in json_files:
        rows.extend(flatten_report(p))

    df_raw = pd.DataFrame(rows)
    if df_raw.empty:
        raise SystemExit("Parsed 0 Korea rows. Check --tag korea / data_path / KOREA_KEY.")

    # 1) дедупликация
    df, df_dups = deduplicate_runs(df_raw)

    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) Все прогоны (после дедупликации)
    out_runs = out_dir / "korea_summary_runs_from_json.csv"
    df.sort_values(["run_stage", "metric_source", "seed", "model"]).to_csv(out_runs, index=False)

    # 2b) Отдельно лог удалённых дублей (очень полезно)
    out_dups = out_dir / "korea_dropped_duplicates.csv"
    df_dups.to_csv(out_dups, index=False)

    # 3) Лидерборд по seed (с metric_source)
    lb = make_leaderboard(df)
    out_lb = out_dir / "korea_leaderboard.csv"
    lb.to_csv(out_lb, index=False)

    # 4) Компактная best-table
    head = make_headline(lb)
    out_head = out_dir / "korea_headline_best.csv"
    head.to_csv(out_head, index=False)

    # 5) Матрица сценарий × модель
    mx = make_scenario_model_matrix(lb)
    out_mx = out_dir / "korea_scenario_model_matrix.csv"
    mx.to_csv(out_mx, index=False)

    # Console preview
    print("[OK] runs table:", out_runs)
    print("[OK] dropped duplicates:", out_dups)
    print("[OK] leaderboard:", out_lb)
    print("[OK] headline:", out_head)
    print("[OK] matrix:", out_mx)
    print()

    print(f"Rows parsed: {len(df_raw)}")
    print(f"Rows kept after dedup: {len(df)}")
    print(f"Duplicates dropped: {len(df_dups)}")
    print()

    print("=== Parsed rows (preview) ===")
    preview_cols = ["run_stage", "metric_source", "seed", "protocol", "sg_deriv", "scenario", "model", "auc", "f1", "recall", "spec"]
    print(df[preview_cols].head(12).to_string(index=False))

    if not head.empty:
        print("\n=== Headline (best per scenario) ===")
        show_cols = ["metric_source", "preproc", "scenario", "best_model", "auc_mean_pm_std", "f1_mean_pm_std", "recall_mean_pm_std", "spec_mean_pm_std"]
        print(head[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
