from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
REPORTS = ROOT / "reports"
FINAL = REPORTS / "final"
FINAL.mkdir(parents=True, exist_ok=True)

COVID_SUMMARY = REPORTS / "summary_grouped.csv"
DIAB_SUMMARY = REPORTS / "summary_grouped_holdout_mean.csv"

OUT_COVID = FINAL / "covid_saliva_supervised_deltas.csv"
OUT_DIAB = FINAL / "diabetes_saliva_supervised_deltas.csv"
OUT_ALL = FINAL / "saliva_supervised_deltas.csv"
OUT_MD = FINAL / "saliva_supervised_summary.md"


METRIC_ALIASES = {
    "auc": ["auc_mean", "roc_auc_mean", "auc", "roc_auc"],
    "pr_auc": ["pr_auc_mean", "prauc_mean", "average_precision_mean", "pr_auc", "prauc", "average_precision"],
    "f1": ["f1_mean", "F1_mean", "f1", "F1"],
    "recall": ["recall_mean", "rec_mean", "recall", "rec"],
    "specificity": ["specificity_mean", "spec_mean", "specificity", "spec"],
    "precision": ["precision_mean", "prec_mean", "precision", "prec"],
    "brier": ["brier_mean", "brier_score_mean", "brier", "brier_score"],
    "ece": ["ece_mean", "ece"],
}


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing input: {path}")
        return pd.DataFrame()
    if path.stat().st_size <= 4:
        print(f"[WARN] Empty or almost empty input: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def infer_dataset_from_scenario(scenario: str) -> str:
    s = str(scenario).lower()

    if "covid" in s or s.startswith("a1_") or s.startswith("a2_"):
        return "covid_saliva"

    if "diab" in s or s.startswith("b1_") or s.startswith("b2_"):
        return "diabetes_saliva"

    return ""


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    if "scenario" not in df.columns:
        raise ValueError("Input summary has no 'scenario' column")

    if "model" not in df.columns:
        raise ValueError("Input summary has no 'model' column")

    if "dataset" not in df.columns:
        df["dataset"] = ""

    inferred = df["scenario"].map(infer_dataset_from_scenario)


    df["dataset_norm"] = df["dataset"].astype("string").fillna("")
    df.loc[df["dataset_norm"].str.strip().eq(""), "dataset_norm"] = inferred

    return df


def metric_col(df: pd.DataFrame, metric: str):
    for col in METRIC_ALIASES[metric]:
        if col in df.columns:
            return col
    return None


def scenario_kind(scenario: str) -> str:
    s = str(scenario).lower()

    if "baseline" in s or s.startswith("a1_") or s.startswith("b1_"):
        return "baseline"

    if "aug" in s or "classic" in s or "strong" in s:
        return "augmented"

    return "other"


def make_deltas(df: pd.DataFrame, dataset: str, protocol_label: str) -> pd.DataFrame:
    df = normalize_df(df)
    if df.empty:
        return pd.DataFrame()

    sub = df[df["dataset_norm"].eq(dataset)].copy()
    if sub.empty:
        print(f"[WARN] No rows for dataset={dataset}")
        print("Available scenarios:")
        print(df[["scenario", "model"]].drop_duplicates().to_string(index=False))
        return pd.DataFrame()

    sub["kind"] = sub["scenario"].map(scenario_kind)

    baseline = sub[sub["kind"].eq("baseline")].copy()
    augmented = sub[sub["kind"].eq("augmented")].copy()

    if baseline.empty:
        print(f"[WARN] No baseline scenario for {dataset}")
        print(sub[["scenario", "model"]].drop_duplicates().to_string(index=False))
        return pd.DataFrame()

    if augmented.empty:
        print(f"[WARN] No augmented scenario for {dataset}")
        print(sub[["scenario", "model"]].drop_duplicates().to_string(index=False))
        return pd.DataFrame()

    rows = []

    for aug_scenario in sorted(augmented["scenario"].dropna().unique()):
        aug_part = augmented[augmented["scenario"].eq(aug_scenario)]

        for model in sorted(set(baseline["model"]).intersection(set(aug_part["model"]))):
            base_rows = baseline[baseline["model"].eq(model)]
            aug_rows = aug_part[aug_part["model"].eq(model)]

            if base_rows.empty or aug_rows.empty:
                continue


            base_row = base_rows.sort_values("scenario").iloc[0]
            aug_row = aug_rows.sort_values("scenario").iloc[0]

            row = {
                "dataset": dataset,
                "protocol": protocol_label,
                "model": model,
                "baseline_scenario": base_row["scenario"],
                "augmented_scenario": aug_row["scenario"],
            }

            if "run_root" in df.columns:
                row["run_root"] = aug_row.get("run_root", "")

            for metric in METRIC_ALIASES:
                col = metric_col(df, metric)
                if col is None:
                    continue

                base_val = pd.to_numeric(base_row.get(col), errors="coerce")
                aug_val = pd.to_numeric(aug_row.get(col), errors="coerce")

                row[f"{metric}_baseline"] = base_val
                row[f"{metric}_augmented"] = aug_val
                row[f"delta_{metric}"] = aug_val - base_val

            rows.append(row)

    out = pd.DataFrame(rows)
    return out


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    cols = [
        "dataset",
        "model",
        "delta_pr_auc",
        "delta_recall",
        "delta_f1",
        "delta_specificity",
        "delta_brier",
        "delta_ece",
    ]
    cols = [c for c in cols if c in df.columns]

    view = df[cols].copy()

    for c in view.columns:
        if c.startswith("delta_"):
            view[c] = pd.to_numeric(view[c], errors="coerce").round(4)

    try:
        return view.to_markdown(index=False)
    except Exception:
        return view.to_string(index=False)


def main():
    covid_df = read_csv_safe(COVID_SUMMARY)
    diab_df = read_csv_safe(DIAB_SUMMARY)

    covid_delta = make_deltas(
        covid_df,
        dataset="covid_saliva",
        protocol_label="MCDCV / grouped by ID",
    )

    diab_delta = make_deltas(
        diab_df,
        dataset="diabetes_saliva",
        protocol_label="CV holdout",
    )

    all_delta = pd.concat([covid_delta, diab_delta], ignore_index=True)

    covid_delta.to_csv(OUT_COVID, index=False)
    diab_delta.to_csv(OUT_DIAB, index=False)
    all_delta.to_csv(OUT_ALL, index=False)

    md = []
    md.append("# Final supervised saliva reports")
    md.append("")
    md.append("This report compares baseline runs with train-only augmentation runs for saliva datasets.")
    md.append("")
    md.append("Positive ΔPR-AUC, ΔRecall and ΔF1 indicate improvement. Negative ΔBrier and ΔECE indicate better calibration.")
    md.append("")
    md.append("## COVID saliva")
    md.append("")
    md.append(md_table(covid_delta))
    md.append("")
    md.append("## Diabetes saliva")
    md.append("")
    md.append(md_table(diab_delta))
    md.append("")
    md.append("## Combined saliva summary")
    md.append("")
    md.append(md_table(all_delta))
    md.append("")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"[OK] Saved: {OUT_COVID} rows={len(covid_delta)}")
    print(f"[OK] Saved: {OUT_DIAB} rows={len(diab_delta)}")
    print(f"[OK] Saved: {OUT_ALL} rows={len(all_delta)}")
    print(f"[OK] Saved: {OUT_MD}")


if __name__ == "__main__":
    main()