from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FINAL = Path("reports/final")
FIGS = Path("reports/figs")
FIGS.mkdir(parents=True, exist_ok=True)


def savefig(name):
    png = FIGS / f"{name}.png"
    pdf = FIGS / f"{name}.pdf"
    plt.tight_layout()
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {png}")
    print(f"[OK] Saved: {pdf}")


def safe_read(path):
    path = Path(path)
    if not path.exists():
        print(f"[SKIP] Missing: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[SKIP] Cannot read {path}: {e}")
        return None


# ---------------------------------------------------------------------
# 1) Saliva: mean deltas by dataset
# ---------------------------------------------------------------------
saliva = safe_read(FINAL / "saliva_supervised_deltas.csv")

if saliva is not None and len(saliva) > 0:
    metrics = [
        "delta_pr_auc",
        "delta_recall",
        "delta_f1",
        "delta_specificity",
        "delta_brier",
        "delta_ece",
    ]
    metrics = [m for m in metrics if m in saliva.columns]

    mean_df = saliva.groupby("dataset")[metrics].mean()

    ax = mean_df.plot(kind="bar", figsize=(11, 5))
    ax.axhline(0, linewidth=1)
    ax.set_title("Saliva datasets: mean effect of train-only augmentation")
    ax.set_ylabel("Augmented - baseline")
    ax.set_xlabel("")
    ax.legend(title="Metric", ncols=3, fontsize=8)
    savefig("fig_saliva_mean_deltas")

    # heatmap by dataset + model
    heat = saliva.copy()
    heat["row"] = heat["dataset"].astype(str) + " / " + heat["model"].astype(str)
    heat = heat.set_index("row")[metrics]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(heat))))
    im = ax.imshow(heat.values, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_title("Saliva supervised deltas by model")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.iloc[i, j]:.3f}", ha="center", va="center", fontsize=7)

    savefig("fig_saliva_model_delta_heatmap")


# ---------------------------------------------------------------------
# 2) GDB dimdesc windows: if possible, make a compact window plot
# ---------------------------------------------------------------------
dimdesc = safe_read(FINAL / "gdb_dimdesc_window_summary.csv")

if dimdesc is not None and len(dimdesc) > 0:
    lower_cols = {c.lower(): c for c in dimdesc.columns}

    window_col = None
    for cand in ["window", "profile", "spectral_window", "range"]:
        if cand in lower_cols:
            window_col = lower_cols[cand]
            break

    delta_cols = [
        c for c in dimdesc.columns
        if "delta" in c.lower() and ("r2" in c.lower() or "max" in c.lower())
    ]

    if window_col and delta_cols:
        delta_col = delta_cols[0]
        plot_df = dimdesc.groupby(window_col)[delta_col].mean().sort_values(ascending=False)

        ax = plot_df.plot(kind="bar", figsize=(8, 4))
        ax.axhline(0, linewidth=1)
        ax.set_title("GDB small-n: geometry effect by spectral window")
        ax.set_ylabel(delta_col)
        ax.set_xlabel("")
        savefig("fig_gdb_dimdesc_window_delta")
    else:
        print("[SKIP] Could not infer columns for GDB dimdesc window plot.")
        print("Columns:", list(dimdesc.columns))


# ---------------------------------------------------------------------
# 3) GDB QC: method-level plots for Amide III and broad window
# ---------------------------------------------------------------------
def make_qc_plot(csv_name, fig_name, title):
    df = safe_read(FINAL / csv_name)
    if df is None or len(df) == 0:
        return

    # Find method column
    method_col = None
    for cand in ["method", "scenario", "generator", "augmentation", "aug_method"]:
        for col in df.columns:
            if col.lower() == cand:
                method_col = col
                break
        if method_col:
            break

    if method_col is None:
        object_cols = [c for c in df.columns if df[c].dtype == "object"]
        if object_cols:
            method_col = object_cols[0]

    # Find QC metric columns
    metric_cols = [
        c for c in df.columns
        if any(x in c.lower() for x in ["auc", "knn", "wasser"])
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not method_col or not metric_cols:
        print(f"[SKIP] Could not infer QC columns in {csv_name}")
        print("Columns:", list(df.columns))
        return

    q = df.groupby(method_col)[metric_cols].mean()

    ax = q.plot(kind="bar", figsize=(10, 5))
    ax.set_title(title)
    ax.set_ylabel("Metric value")
    ax.set_xlabel("")
    ax.legend(title="QC metric", fontsize=8)
    savefig(fig_name)


make_qc_plot(
    "gdb_qc_amide3_method_summary.csv",
    "fig_gdb_qc_amide3_methods",
    "GDB Amide III: synthetic-data QC by method",
)

make_qc_plot(
    "gdb_qc_broad_method_summary.csv",
    "fig_gdb_qc_broad_methods",
    "GDB broad window: synthetic-data QC by method",
)

print("\nDONE. Figures are in reports/figs/")
