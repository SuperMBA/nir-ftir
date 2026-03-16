#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot dimdesc-like R² (eta²) curves across PCs: baseline vs classic (aug).
Input: reports/pca_r2/dimdesc_r2_summary.csv (made by aggregate_dimdesc_r2.py)
Output: PNGs in --outdir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pc_to_int(pc: str) -> int:
    # expects "PC1", "PC10", ...
    s = str(pc).strip().upper().replace(" ", "")
    if not s.startswith("PC"):
        raise ValueError(f"Bad pc label: {pc}")
    return int(s[2:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary-csv",
        default="reports/pca_r2/dimdesc_r2_summary.csv",
        help="CSV produced by scripts/aggregate_dimdesc_r2.py",
    )
    ap.add_argument("--preproc", default="amide3", choices=["amide3", "paper_full", "paper_low"])
    ap.add_argument("--use-mixup", type=int, default=0)
    ap.add_argument(
        "--factors",
        default="",
        help="Comma-separated list (optional). If empty -> all factors in the CSV for selected preproc.",
    )
    ap.add_argument("--outdir", default="reports/pca_r2/figs")
    ap.add_argument("--max-pc", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    df = df[(df["preproc"] == args.preproc) & (df["use_mixup"] == args.use_mixup)].copy()
    if df.empty:
        raise SystemExit(f"No rows for preproc={args.preproc}, use_mixup={args.use_mixup} in {args.summary_csv}")

    df["pc_i"] = df["pc"].apply(pc_to_int)
    df = df[df["pc_i"] <= args.max_pc].copy()

    if args.factors.strip():
        factors = [x.strip() for x in args.factors.split(",") if x.strip()]
    else:
        factors = sorted(df["factor"].unique().tolist())

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Print quick “best PC” per factor
    best = (
        df.sort_values(["factor", "delta_r2_mean"], ascending=[True, False])
        .groupby("factor", as_index=False)
        .head(1)
        .sort_values("delta_r2_mean", ascending=False)
    )
    print("\n[Best PC per factor] (within max_pc):")
    print(best[["factor", "pc", "r2_baseline_mean", "r2_aug_mean", "delta_r2_mean", "delta_r2_std", "n"]].to_string(index=False))

    for fac in factors:
        sub = df[df["factor"] == fac].copy()
        if sub.empty:
            print(f"[WARN] factor not found in filtered data: {fac}")
            continue

        # Make sure sorted by PC
        sub = sub.sort_values("pc_i")

        x = sub["pc_i"].to_numpy()
        b = sub["r2_baseline_mean"].to_numpy()
        a = sub["r2_aug_mean"].to_numpy()
        # std (optional)
        bstd = sub.get("r2_baseline_std", pd.Series([np.nan] * len(sub))).to_numpy()
        astd = sub.get("r2_aug_std", pd.Series([np.nan] * len(sub))).to_numpy()

        plt.figure()
        plt.plot(x, b, marker="o", label="baseline")
        plt.plot(x, a, marker="o", label="classic_aug")

        # shade if std exists (may be NaN)
        if np.isfinite(bstd).any():
            plt.fill_between(x, b - bstd, b + bstd, alpha=0.15)
        if np.isfinite(astd).any():
            plt.fill_between(x, a - astd, a + astd, alpha=0.15)

        # highlight best delta PC for this factor
        i_best = int(sub["delta_r2_mean"].idxmax())
        pc_best = int(df.loc[i_best, "pc_i"])
        plt.axvline(pc_best, linestyle="--", linewidth=1)

        plt.title(f"dimdesc-like R² across PCs | {args.preproc} | {fac}")
        plt.xlabel("PC index")
        plt.ylabel("R² (eta²)")
        plt.xticks(x)
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.25)
        plt.legend()

        fname = outdir / f"r2_curve_{args.preproc}_mix{args.use_mixup}_{fac}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=args.dpi)
        plt.close()
        print(f"[OK] saved {fname}")

    print("\nDone.")


if __name__ == "__main__":
    main()