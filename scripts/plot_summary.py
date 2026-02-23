# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "reports" / "summary_runs_from_json.csv"
OUTD = ROOT / "reports" / "plots"
OUTD.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INP)

# берем только holdout (по seed красиво)
h = df[df["metric_source"]=="holdout_test"].copy()

def plot_metric(metric: str, dataset: str):
    d = h[h["dataset"]==dataset].copy()
    if d.empty:
        return
    # (scenario, model) -> список значений по seed
    keys = sorted(d[["scenario","model"]].drop_duplicates().itertuples(index=False, name=None))
    data = [d[(d["scenario"]==sc)&(d["model"]==m)][metric].values for sc,m in keys]
    labels = [f"{sc}\n{m}" for sc,m in keys]

    plt.figure(figsize=(max(8, 0.7*len(labels)), 5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel(metric)
    plt.title(f"{dataset}: {metric} across seeds (holdout)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTD / f"{dataset}_{metric}_box.png", dpi=200)
    plt.close()

for ds in sorted(h["dataset"].unique()):
    for met in ["f1","auc","spec","recall"]:
        plot_metric(met, ds)

print("[OK] plots in:", OUTD)
