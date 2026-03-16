# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

METRIC_COLS = ["auc", "pr_auc", "f1", "recall", "prec", "spec", "acc", "brier", "ece"]

# служебные json-ы, которые точно не являются отчётами метрик
SKIP_FILENAMES = {
    "config.json",
    "manifest.json",
}


# ------------------------------
# Utils
# ------------------------------
def detect_root() -> Path:
    """Пытается найти корень репозитория (где есть src/ и reports/)."""
    here = Path(__file__).resolve()
    candidates = [here.parent, here.parent.parent]
    for c in candidates:
        if (c / "reports").exists() and (c / "src").exists():
            return c
    return here.parent


ROOT = detect_root()


def _f(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, str) and x.strip() == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def safe_read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _metric_get(d: dict, key: str) -> float:
    """Поддержка нескольких вариантов имён метрик."""
    aliases = {
        "auc": ["auc", "roc_auc"],
        "pr_auc": ["pr_auc", "auprc", "average_precision", "ap"],
        "f1": ["f1", "f1_score"],
        "recall": ["recall", "sens", "sensitivity", "tpr"],
        "prec": ["prec", "precision", "ppv"],
        "spec": ["spec", "specificity", "tnr"],
        "acc": ["acc", "accuracy"],
        "brier": ["brier", "brier_score"],
        "ece": ["ece"],
    }
    # case-insensitive lookup
    if not isinstance(d, dict):
        return float("nan")
    dl = {str(k).lower(): v for k, v in d.items()}
    for k in aliases.get(key, [key]):
        kl = str(k).lower()
        if kl in dl:
            return _f(dl.get(kl))
    return float("nan")


def _nan_cv(std_val: float, mean_val: float) -> float:
    if any(
        map(
            lambda z: z is None or (isinstance(z, float) and math.isnan(z)),
            [std_val, mean_val],
        )
    ):
        return float("nan")
    if mean_val == 0:
        return float("nan")
    return float(std_val) / abs(float(mean_val))


def _is_report_json(path: Path) -> bool:
    if path.suffix.lower() != ".json":
        return False
    if path.name in SKIP_FILENAMES:
        return False
    return True


# ------------------------------
# Path/scenario parsing
# ------------------------------
@dataclass
class PathInfo:
    run_name: str
    scenario: str
    seed_dir: str
    task: str
    aug_kind: str
    is_meta: int
    phase_code: str


def parse_path_info(json_path: Path, reports_root: Path) -> PathInfo:
    rel = json_path.resolve().relative_to(reports_root.resolve())
    parts = rel.parts

    run_name = parts[0] if len(parts) >= 1 else "unknown_run"
    # typical: <run_name>/<scenario>/seed_<n>/<file>.json
    scenario = parts[1] if len(parts) >= 2 else json_path.parent.name
    seed_dir = parts[2] if len(parts) >= 3 else json_path.parent.name

    scenario_l = scenario.lower()
    phase_code = ""
    m = re.match(r"^(p\d+[a-z]?)_", scenario_l)
    if m:
        phase_code = m.group(1)

    # task
    if "parodont" in scenario_l:
        task = "parodont_H_vs_path"
    elif "anamnes" in scenario_l:
        task = "anamnes_H_vs_path"
    elif "healthy_any" in scenario_l or "stress" in scenario_l:
        task = "healthy_vs_any"
    elif "caries" in scenario_l:
        task = "caries"
    else:
        task = "unknown"

    # augmentation kind
    if "classic_aug" in scenario_l:
        aug_kind = "classic_aug"
    elif "vae" in scenario_l:
        aug_kind = "vae_aug"
    elif "wgan" in scenario_l:
        aug_kind = "wgan_aug"
    elif "baseline" in scenario_l:
        aug_kind = "baseline"
    else:
        aug_kind = "other"

    is_meta = 1 if ("_meta" in scenario_l or scenario_l.endswith("m") or "meta" in scenario_l) else 0

    return PathInfo(
        run_name=run_name,
        scenario=scenario,
        seed_dir=seed_dir,
        task=task,
        aug_kind=aug_kind,
        is_meta=is_meta,
        phase_code=phase_code,
    )


def _infer_label_col(task: str) -> str:
    if task == "parodont_H_vs_path":
        return "y_parodont_H_vs_path"
    if task == "anamnes_H_vs_path":
        return "y_anamnes_H_vs_path"
    if task == "healthy_vs_any":
        return "y_healthy_vs_any"
    return ""


def _infer_seed(cfg: dict, obj: dict, pinfo: PathInfo, json_path: Path) -> Optional[int]:
    seed = cfg.get("seed", obj.get("seed", None))
    if seed is not None:
        try:
            return int(seed)
        except Exception:
            pass

    # seed from folder name seed_0
    m = re.search(r"seed_(\d+)", str(pinfo.seed_dir))
    if m:
        return int(m.group(1))

    # seed from filename ..._seed_0_... or ...seed0...
    m = re.search(r"(?:^|_)seed[_-]?(\d+)(?:_|\.|$)", json_path.name)
    if m:
        return int(m.group(1))

    return None


# ------------------------------
# JSON summary extraction (robust)
# ------------------------------
def _iter_candidate_dicts(obj: dict) -> Iterable[dict]:
    """Выдаёт кандидаты словарей, где потенциально лежат метрики."""
    if not isinstance(obj, dict):
        return
    # top-level common keys
    for k in ("summary", "mcdcv", "mcdcv_summary", "cv_summary", "cv", "metrics_summary", "metrics"):
        v = obj.get(k)
        if isinstance(v, dict):
            yield v
    # nested common pattern: {"mcdcv": {"summary": {...}}}
    for k in ("summary", "mcdcv", "cv", "results"):
        v = obj.get(k)
        if isinstance(v, dict):
            for kk in ("summary", "mcdcv", "cv", "metrics"):
                vv = v.get(kk)
                if isinstance(vv, dict):
                    yield vv


def _looks_like_mean_std_block(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    if "mean" in d and isinstance(d["mean"], dict):
        return True
    return False


def _split_mean_std(metrics_block: dict) -> Tuple[dict, dict]:
    """
    Нормализует разные форматы:
    - {"mean": {...}, "std": {...}}
    - {"auc":..., "auc_std":..., ...}
    - {"mean": {...}} (без std)
    - {"metrics": {...}, "metrics_std": {...}} (редко)
    """
    if not isinstance(metrics_block, dict):
        return {}, {}

    # canonical
    if isinstance(metrics_block.get("mean"), dict):
        mean_d = metrics_block.get("mean") or {}
        std_d = metrics_block.get("std") if isinstance(metrics_block.get("std"), dict) else {}
        return mean_d, std_d

    # flat with *_std
    mean_d: dict = {}
    std_d: dict = {}
    for k, v in metrics_block.items():
        if not isinstance(k, str):
            continue
        if k.endswith("_std"):
            std_d[k[:-4]] = v
        else:
            mean_d[k] = v

    # if there is a nested std dict, merge it
    if isinstance(metrics_block.get("std"), dict):
        for k, v in (metrics_block.get("std") or {}).items():
            std_d[str(k)] = v

    return mean_d, std_d


def _extract_mcdcv_rows(summary_like: dict, common: dict) -> List[dict]:
    """
    Пытается извлечь агрегированные метрики (mean/std) по моделям.
    Поддерживает:
      A) {"mean": {"plsda": {...}}, "std": {"plsda": {...}}}
      B) {"plsda": {"auc":..., "auc_std":...}}
      C) {"models": {"plsda": {...}}} (редко)
    """
    rows: List[dict] = []
    if not isinstance(summary_like, dict):
        return rows

    # case A: has mean dict keyed by model
    if isinstance(summary_like.get("mean"), dict):
        mean_by_model = summary_like.get("mean") or {}
        std_by_model = summary_like.get("std") if isinstance(summary_like.get("std"), dict) else {}
        for model, mean_metrics in mean_by_model.items():
            if not isinstance(mean_metrics, dict):
                continue
            std_metrics = std_by_model.get(model, {}) if isinstance(std_by_model, dict) else {}
            if not isinstance(std_metrics, dict):
                std_metrics = {}

            row = {
                **common,
                "model": str(model),
                "metric_source": "mcdcv_mean",
                "thr": float("nan"),
                "tp": float("nan"),
                "fp": float("nan"),
                "tn": float("nan"),
                "fn": float("nan"),
            }
            for k in METRIC_COLS:
                row[k] = _metric_get(mean_metrics, k)
                row[f"{k}_std"] = _metric_get(std_metrics, k)
            rows.append(row)
        return rows

    # case B/C: dict keyed by models directly
    # (например: {"plsda": {"auc":..., "auc_std":...}, "svm": {...}})
    # если внутри есть служебные ключи, отфильтруем их
    skip_keys = {"std", "mean", "n", "count", "meta"}
    candidates = {k: v for k, v in summary_like.items() if k not in skip_keys and isinstance(v, dict)}
    # иногда "models": {...}
    if not candidates and isinstance(summary_like.get("models"), dict):
        candidates = {k: v for k, v in (summary_like.get("models") or {}).items() if isinstance(v, dict)}

    for model, block in candidates.items():
        mean_metrics, std_metrics = _split_mean_std(block)

        # must contain at least one known metric to be considered a report
        if not any(_metric_get(mean_metrics, m) == _metric_get(mean_metrics, m) for m in METRIC_COLS):
            # (nan != nan) trick: above checks "is not nan"
            pass

        # require at least one metric key present (raw key check is safer)
        present = any(
            k in {str(x).lower() for x in mean_metrics.keys()}
            for k in ("auc", "roc_auc", "f1", "recall", "precision", "spec", "pr_auc", "average_precision", "brier", "ece")
        )
        if not present:
            continue

        row = {
            **common,
            "model": str(model),
            "metric_source": "mcdcv_mean",
            "thr": float("nan"),
            "tp": float("nan"),
            "fp": float("nan"),
            "tn": float("nan"),
            "fn": float("nan"),
        }
        for k in METRIC_COLS:
            row[k] = _metric_get(mean_metrics, k)
            row[f"{k}_std"] = _metric_get(std_metrics, k)
        rows.append(row)

    return rows


def _extract_holdout_rows(results_like: Any, common: dict) -> List[dict]:
    """
    Извлекает holdout-like результаты:
      A) {"plsda": {"test": {...}}}
      B) {"plsda": {"test": {"metrics": {...}}}}
      C) {"test": {...}} (single-model), тогда model берём из common['models_cfg'] или report filename не знаем -> "model"
    """
    rows: List[dict] = []
    if not isinstance(results_like, dict):
        return rows

    def _row_for(model: str, t: dict) -> dict:
        row = {
            **common,
            "model": model,
            "metric_source": "holdout_test",
            "thr": _f(t.get("thr", t.get("threshold"))),
            "tp": _f(t.get("tp")),
            "fp": _f(t.get("fp")),
            "tn": _f(t.get("tn")),
            "fn": _f(t.get("fn")),
        }
        for k in METRIC_COLS:
            row[k] = _metric_get(t, k)
            row[f"{k}_std"] = float("nan")
        return row

    # A/B
    any_model_keys = False
    for model, r in results_like.items():
        if not isinstance(r, dict):
            continue
        # skip service keys
        if str(model).lower() in {"config", "summary", "meta"}:
            continue
        t = r.get("test", {}) or {}
        if isinstance(t, dict) and isinstance(t.get("metrics"), dict):
            t = t["metrics"]
        if not isinstance(t, dict):
            continue

        present = any(
            k in {str(x).lower() for x in t.keys()}
            for k in ("auc", "roc_auc", "f1", "recall", "precision", "prec", "spec", "accuracy", "acc", "pr_auc", "average_precision")
        )
        if not present:
            continue

        any_model_keys = True
        rows.append(_row_for(str(model), t))

    if any_model_keys:
        return rows

    # C) single-model
    t = results_like.get("test", {}) or {}
    if isinstance(t, dict) and isinstance(t.get("metrics"), dict):
        t = t["metrics"]
    if isinstance(t, dict):
        present = any(
            k in {str(x).lower() for x in t.keys()}
            for k in ("auc", "roc_auc", "f1", "recall", "precision", "prec", "spec", "accuracy", "acc", "pr_auc", "average_precision")
        )
        if present:
            rows.append(_row_for("model", t))

    return rows


# ------------------------------
# JSON flattening
# ------------------------------
def flatten_report(json_path: Path, reports_root: Path) -> List[dict]:
    obj = safe_read_json(json_path)
    if not obj:
        return []

    cfg = obj.get("config", {}) or {}
    detected = obj.get("detected", {}) or {}

    pinfo = parse_path_info(json_path, reports_root)

    # dataset/protocol
    dataset = cfg.get("dataset", obj.get("dataset", ""))
    protocol = obj.get("protocol", cfg.get("protocol", ""))

    # seed/label_col
    seed = _infer_seed(cfg, obj, pinfo, json_path)
    label_col = cfg.get("label_col", obj.get("label_col", "")) or _infer_label_col(pinfo.task)

    common = {
        "json_path": str(json_path),
        "report_file": json_path.name,
        "run_dir": str(json_path.parent),
        "report_mtime": json_path.stat().st_mtime,
        "run_name": pinfo.run_name,
        "scenario": pinfo.scenario,
        "seed_dir": pinfo.seed_dir,
        "phase_code": pinfo.phase_code,
        "task": pinfo.task,
        "aug_kind": pinfo.aug_kind,
        "is_meta": pinfo.is_meta,
        "dataset": dataset,
        "protocol": protocol,
        "seed": seed,
        "label_col": label_col,
        "threshold_by": cfg.get("threshold_by"),
        "recall_target": _f(cfg.get("recall_target")),
        "min_spec": _f(cfg.get("min_spec")),
        "min_prec": _f(cfg.get("min_prec")),
        "group_col": cfg.get("group_col", detected.get("group_col")),
        "n_splits": _f(cfg.get("n_splits")),
        "val_size": _f(cfg.get("val_size")),
        "calib": cfg.get("calib"),
        "calib_real_only": int(bool(cfg.get("calib_real_only"))) if "calib_real_only" in cfg else np.nan,
        "calib_frac": _f(cfg.get("calib_frac")),
        "meta_stratify": cfg.get("meta_stratify", ""),
        "crop_min": _f(cfg.get("crop_min")),
        "crop_max": _f(cfg.get("crop_max")),
        "sg_window": _f(cfg.get("sg_window")),
        "sg_poly": _f(cfg.get("sg_poly")),
        "sg_deriv": _f(cfg.get("sg_deriv")),
        "norm": cfg.get("norm"),
        "xscale": cfg.get("xscale"),
        "noise_std": _f(cfg.get("noise_std")),
        "noise_med": _f(cfg.get("noise_med")),
        "shift": _f(cfg.get("shift")),
        "scale": _f(cfg.get("scale")),
        "tilt": _f(cfg.get("tilt")),
        "offset": _f(cfg.get("offset")),
        "mixup": _f(cfg.get("mixup")),
        "mixwithin": _f(cfg.get("mixwithin")),
        "aug_repeats": _f(cfg.get("aug_repeats")),
        "p_apply": _f(cfg.get("p_apply")),
        "models_cfg": cfg.get("models"),
        "n_groups": _f(detected.get("n_groups")),
        "max_reps_per_group": _f(detected.get("max_reps_per_group")),
        "n_samples": _f(detected.get("n_samples")),
        "n_features": _f(detected.get("n_features")),
        "pos_rate_detected": _f(detected.get("pos_rate")),
    }

    rows: List[dict] = []

    # summary-like candidates (mcdcv mean/std)
    for cand in _iter_candidate_dicts(obj):
        rows.extend(_extract_mcdcv_rows(cand, common))

    # holdout-like
    results_like = obj.get("results") or obj.get("holdout") or {}
    if isinstance(results_like, dict) and isinstance(results_like.get("results"), dict):
        results_like = results_like["results"]
    rows.extend(_extract_holdout_rows(results_like, common))

    return rows


# ------------------------------
# Aggregations
# ------------------------------
def dedupe_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Убирает дубликаты прогонов (одинаковые scenario+seed+model+metric_source+label_col).
    Оставляет самый свежий JSON по mtime.
    """
    if df.empty:
        return df.copy()

    # ВАЖНО: run_name НЕ включаем, чтобы схлопывать повторные перезапуски того же сценария
    # в разных папках (берём самый свежий JSON).
    key_cols = [
        "scenario",
        "seed",
        "model",
        "metric_source",
        "label_col",
        "protocol",
        "dataset",
        "n_splits",
        "val_size",
        "calib",
        "calib_frac",
        "threshold_by",
        "recall_target",
        "min_spec",
        "min_prec",
        "meta_stratify",
        "noise_std",
        "noise_med",
        "shift",
        "scale",
        "tilt",
        "offset",
        "mixup",
        "mixwithin",
        "aug_repeats",
        "p_apply",
    ]
    for c in key_cols:
        if c not in df.columns:
            df[c] = np.nan

    d = df.copy()
    d = d.sort_values([*key_cols, "report_mtime", "json_path"], ascending=[True] * len(key_cols) + [False, False])
    d = d.drop_duplicates(subset=key_cols, keep="first")
    return d


def build_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    grp_cols = [
        "run_name",
        "task",
        "scenario",
        "phase_code",
        "aug_kind",
        "is_meta",
        "label_col",
        "metric_source",
        "model",
        "dataset",
        "protocol",
        "threshold_by",
        "calib",
        "n_splits",
    ]

    agg_spec: Dict[str, List[str]] = {}
    for m in METRIC_COLS:
        agg_spec[m] = ["mean", "std", "median", "min", "max", "count"]
    for c in ["thr", "tp", "fp", "tn", "fn"]:
        agg_spec[c] = ["mean", "std"]

    g = df.groupby(grp_cols, dropna=False).agg(agg_spec)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reset_index()

    # stability helpers
    for m in ["f1", "recall", "spec", "pr_auc", "auc", "brier", "ece"]:
        mean_c = f"{m}_mean"
        std_c = f"{m}_std"
        cv_c = f"{m}_cv"
        if mean_c in g.columns and std_c in g.columns:
            g[cv_c] = [_nan_cv(_f(s), _f(mu)) for s, mu in zip(g[std_c].tolist(), g[mean_c].tolist())]

    # primary sort score (task-aware)
    g["primary_metric"] = np.where(g["task"].eq("healthy_vs_any"), "recall", "f1")
    g["sort_score"] = np.where(g["task"].eq("healthy_vs_any"), g.get("recall_mean", np.nan), g.get("f1_mean", np.nan))

    return g


def add_leaderboard_ranks(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()
    df = summary_df.copy()

    def _rank_block(block: pd.DataFrame) -> pd.DataFrame:
        task_val = str(block["task"].iloc[0]) if len(block) else ""
        if task_val == "healthy_vs_any":
            sort_cols = ["recall_mean", "f1_mean", "spec_mean", "pr_auc_mean"]
        else:
            sort_cols = ["f1_mean", "pr_auc_mean", "recall_mean", "spec_mean"]
        for c in sort_cols:
            if c not in block.columns:
                block[c] = np.nan
        block = block.sort_values(sort_cols, ascending=[False] * len(sort_cols)).copy()
        block["rank_in_scenario"] = range(1, len(block) + 1)
        return block

    ranked = (
        df.groupby(["run_name", "scenario", "metric_source"], dropna=False, group_keys=False)
        .apply(_rank_block)
        .reset_index(drop=True)
    )
    return ranked


def build_aug_pairs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Парный baseline vs classic_aug по одному и тому же seed/model/task/scenario."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    x = df[df["aug_kind"].isin(["baseline", "classic_aug"])].copy()
    x = x[x["is_meta"].fillna(0).astype(float) == 0].copy()
    if x.empty:
        return pd.DataFrame(), pd.DataFrame()

    # IMPORTANT: include scenario to avoid mixing T1/T2 (и вообще разных папок)
    pair_keys = ["run_name", "scenario", "task", "label_col", "metric_source", "model", "seed"]

    keep_cols = pair_keys + ["aug_kind"] + METRIC_COLS + ["thr", "tp", "fp", "tn", "fn"]
    for c in keep_cols:
        if c not in x.columns:
            x[c] = np.nan
    x = x[keep_cols].copy()

    wide = (
        x.pivot_table(
            index=pair_keys,
            columns="aug_kind",
            values=METRIC_COLS + ["thr", "tp", "fp", "tn", "fn"],
            aggfunc="first",
        )
        .reset_index()
    )
    if wide.empty:
        return pd.DataFrame(), pd.DataFrame()

    # flatten columns
    flat_cols: List[str] = []
    for c in wide.columns:
        if isinstance(c, tuple):
            left, right = c
            if right == "":
                flat_cols.append(str(left))
            else:
                flat_cols.append(f"{left}_{right}")
        else:
            flat_cols.append(str(c))
    wide.columns = flat_cols

    # deltas classic - baseline
    for m in METRIC_COLS + ["thr"]:
        cb = f"{m}_classic_aug"
        bb = f"{m}_baseline"
        if cb in wide.columns and bb in wide.columns:
            wide[f"delta_{m}"] = pd.to_numeric(wide[cb], errors="coerce") - pd.to_numeric(wide[bb], errors="coerce")

    delta_cols = [c for c in wide.columns if c.startswith("delta_")]
    if not delta_cols:
        return wide, pd.DataFrame()

    agg_spec = {c: ["mean", "std", "median", "count"] for c in delta_cols}
    agg = wide.groupby(["run_name", "scenario", "task", "label_col", "metric_source", "model"], dropna=False).agg(agg_spec)
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()

    if {"delta_f1_mean", "delta_f1_std"}.issubset(agg.columns):
        agg["delta_f1_cv"] = [_nan_cv(_f(s), _f(mu)) for s, mu in zip(agg["delta_f1_std"].tolist(), agg["delta_f1_mean"].tolist())]
        agg["delta_f1_stability_flag"] = np.where(
            (agg["delta_f1_cv"].abs() > 1.0) | (agg["delta_f1_count"] < 5),
            "unstable",
            "ok",
        )

    return wide, agg


# ------------------------------
# Main
# ------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate JSON reports for GDB/Korea small-n experiments")
    ap.add_argument("--reports-root", default=str(ROOT / "reports" / "exp"), help="Path to reports/exp")
    ap.add_argument("--out-dir", default=str(ROOT / "reports"), help="Where to save CSV summaries")
    ap.add_argument("--run-prefix", default="gdb_smalln_", help="Use only runs with this folder prefix (empty = all)")
    ap.add_argument("--run-name", default="", help="Exact run folder name (overrides --run-prefix)")
    ap.add_argument("--latest-only", action="store_true", help="Take only the latest matching run folder")
    ap.add_argument("--pattern", default="*.json", help="JSON filename glob under selected runs (e.g. '*.json')")
    ap.add_argument("--debug", action="store_true", help="Print brief diagnostics if parsed 0 rows")
    args = ap.parse_args()

    reports_root = Path(args.reports_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not reports_root.exists():
        raise SystemExit(f"Reports dir not found: {reports_root}")

    # select run folders
    run_dirs = [p for p in sorted(reports_root.iterdir()) if p.is_dir()]
    if args.run_name:
        run_dirs = [p for p in run_dirs if p.name == args.run_name]
    elif args.run_prefix:
        run_dirs = [p for p in run_dirs if p.name.startswith(args.run_prefix)]

    if not run_dirs:
        raise SystemExit("No matching run directories found.")

    run_dirs = sorted(run_dirs, key=lambda p: p.stat().st_mtime)
    if args.latest_only:
        run_dirs = [run_dirs[-1]]

    # collect json files
    json_files: List[Path] = []
    for rd in run_dirs:
        json_files.extend(sorted(rd.rglob(args.pattern)))

    json_files = [p for p in json_files if _is_report_json(p)]

    if not json_files:
        raise SystemExit(f"No JSON files matched pattern '{args.pattern}' in selected runs.")

    rows: List[dict] = []
    parsed_files = 0
    for p in json_files:
        r = flatten_report(p, reports_root)
        if r:
            parsed_files += 1
            rows.extend(r)

    df = pd.DataFrame(rows)
    if df.empty:
        if args.debug:
            print("[DEBUG] Parsed 0 rows.")
            print("[DEBUG] JSON files scanned:", len(json_files))
            for p in json_files[:10]:
                obj = safe_read_json(p)
                top_keys = list(obj.keys())[:20] if isinstance(obj, dict) else []
                print(" -", p, "keys:", top_keys)
        raise SystemExit("Parsed 0 rows from JSON files.")

    # normalize types
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")

    # order columns (important ones first)
    ordered = [
        "json_path",
        "report_file",
        "report_mtime",
        "run_dir",
        "run_name",
        "scenario",
        "phase_code",
        "task",
        "aug_kind",
        "is_meta",
        "dataset",
        "protocol",
        "label_col",
        "seed",
        "model",
        "metric_source",
        *METRIC_COLS,
        *[f"{m}_std" for m in METRIC_COLS],
        "thr",
        "tp",
        "fp",
        "tn",
        "fn",
        "threshold_by",
        "recall_target",
        "min_spec",
        "min_prec",
        "group_col",
        "n_splits",
        "val_size",
        "calib",
        "calib_real_only",
        "calib_frac",
        "meta_stratify",
        "crop_min",
        "crop_max",
        "sg_window",
        "sg_poly",
        "sg_deriv",
        "norm",
        "xscale",
        "noise_std",
        "noise_med",
        "shift",
        "scale",
        "tilt",
        "offset",
        "mixup",
        "mixwithin",
        "aug_repeats",
        "p_apply",
        "n_groups",
        "max_reps_per_group",
        "n_samples",
        "n_features",
        "pos_rate_detected",
        "models_cfg",
        "seed_dir",
    ]
    for c in ordered:
        if c not in df.columns:
            df[c] = np.nan
    df = df[ordered + [c for c in df.columns if c not in ordered]]

    df_dedup = dedupe_rows(df)
    seed_summary = build_seed_summary(df_dedup)
    leaderboard = add_leaderboard_ranks(seed_summary)
    aug_pairs, aug_delta = build_aug_pairs(df_dedup)

    # best per scenario (rank 1)
    best_per_scenario = pd.DataFrame()
    if not leaderboard.empty and "rank_in_scenario" in leaderboard.columns:
        best_per_scenario = leaderboard[leaderboard["rank_in_scenario"] == 1].copy()

    # output prefix
    if args.run_name:
        tag = args.run_name
    elif args.latest_only and run_dirs:
        tag = run_dirs[-1].name
    elif args.run_prefix:
        tag = f"{args.run_prefix}ALL".replace("*", "")
    else:
        tag = "all_runs"
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)

    out_runs = out_dir / f"{tag}__summary_runs_raw.csv"
    out_runs_dedup = out_dir / f"{tag}__summary_runs_dedup.csv"
    out_seed = out_dir / f"{tag}__summary_seed_stats.csv"
    out_lb = out_dir / f"{tag}__leaderboard.csv"
    out_best = out_dir / f"{tag}__leaderboard_best_per_scenario.csv"
    out_pairs = out_dir / f"{tag}__aug_pairs_baseline_vs_classic.csv"
    out_delta = out_dir / f"{tag}__aug_delta_summary.csv"

    df.to_csv(out_runs, index=False, encoding="utf-8-sig")
    df_dedup.to_csv(out_runs_dedup, index=False, encoding="utf-8-sig")
    seed_summary.to_csv(out_seed, index=False, encoding="utf-8-sig")
    leaderboard.to_csv(out_lb, index=False, encoding="utf-8-sig")
    best_per_scenario.to_csv(out_best, index=False, encoding="utf-8-sig")
    if not aug_pairs.empty:
        aug_pairs.to_csv(out_pairs, index=False, encoding="utf-8-sig")
    if not aug_delta.empty:
        aug_delta.to_csv(out_delta, index=False, encoding="utf-8-sig")

    # compact console summary
    print("[OK] Reports root:", reports_root)
    print("[OK] Selected run dirs:", len(run_dirs))
    for rd in run_dirs:
        print("   -", rd.name)
    print("[OK] JSON files scanned:", len(json_files))
    print("[OK] Files with parsed metrics:", parsed_files)
    print("[OK] Parsed rows:", len(df))
    print("[OK] Rows after dedupe:", len(df_dedup))
    if len(df) != len(df_dedup):
        print("[INFO] Duplicates removed:", len(df) - len(df_dedup))

    print("[OK] Saved:", out_runs)
    print("[OK] Saved:", out_runs_dedup)
    print("[OK] Saved:", out_seed)
    print("[OK] Saved:", out_lb)
    print("[OK] Saved:", out_best)
    if not aug_pairs.empty:
        print("[OK] Saved:", out_pairs)
    if not aug_delta.empty:
        print("[OK] Saved:", out_delta)

    # preview top-1 rows for convenience
    if not best_per_scenario.empty:
        cols = [c for c in ["scenario", "metric_source", "model", "f1_mean", "recall_mean", "spec_mean", "pr_auc_mean", "auc_mean", "f1_std", "recall_std"] if c in best_per_scenario.columns]
        print("\nTop model per scenario:")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(best_per_scenario[cols].sort_values(["scenario", "metric_source"]).to_string(index=False))


if __name__ == "__main__":
    main()