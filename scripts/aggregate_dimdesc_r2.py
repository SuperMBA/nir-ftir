# scripts/aggregate_dimdesc_r2.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def main():
    base = Path("reports/pca_r2")
    paths = sorted(base.rglob("dimdesc_like_r2.csv"))
    if not paths:
        raise SystemExit("No dimdesc_like_r2.csv found under reports/pca_r2/")

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

    # summary per (preproc, factor, pc)
    grp = df.groupby(["preproc", "use_mixup", "factor", "pc"], as_index=False).agg(
        r2_baseline_mean=("r2_baseline", "mean"),
        r2_aug_mean=("r2_aug", "mean"),
        delta_r2_mean=("delta_r2", "mean"),
        delta_r2_std=("delta_r2", "std"),
        n=("delta_r2", "count"),
        p_aug_frac_lt_0_05=("p_aug", lambda x: float(np.mean(pd.to_numeric(x, errors="coerce") < 0.05))),
    )

    out1 = base / "dimdesc_r2_summary.csv"
    grp.sort_values(["delta_r2_mean"], ascending=False).to_csv(out1, index=False)

    # best PC per factor within each preproc (by delta_r2_mean)
    best = (
        grp.sort_values(["preproc", "use_mixup", "factor", "delta_r2_mean"], ascending=[True, True, True, False])
        .groupby(["preproc", "use_mixup", "factor"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    out2 = base / "dimdesc_r2_best_pc_per_factor.csv"
    best.to_csv(out2, index=False)

    print("[OK] saved:", out1)
    print("[OK] saved:", out2)
    print("\nTop-10 improvements:")
    print(grp.sort_values("delta_r2_mean", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()