from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# === PATHS ===
ROOT = Path(__file__).resolve().parents[1]

PATH_BEST = ROOT / "reports" / "pca_r2" / "dimdesc_r2_best_pc_per_factor.csv"
PATH_SUM = ROOT / "reports" / "pca_r2" / "dimdesc_r2_summary.csv"

OUTDIR = ROOT / "reports" / "figs"
OUTDIR.mkdir(parents=True, exist_ok=True)


def fig1_window_deltas(path_best: Path, use_mixup: int = 0) -> None:
    """Main figure: Δmax R²(best-PC) by spectral window (bars = mean over factors, dots = factors)."""
    df = pd.read_csv(path_best)
    df = df[df["use_mixup"] == use_mixup].copy()

    windows_order = ["paper_full", "paper_low", "amide3"]
    df["preproc"] = pd.Categorical(df["preproc"], categories=windows_order, ordered=True)

    g = df.groupby("preproc")["delta_r2_mean"]
    mean_delta = g.mean().reindex(windows_order)
    std_delta = g.std(ddof=1).reindex(windows_order)

    x = np.arange(len(windows_order))

    plt.figure(figsize=(7.2, 4.2))
    plt.bar(x, mean_delta.values, yerr=std_delta.values, capsize=4)

    # factor points (jitter)
    rng = np.random.default_rng(0)
    for i, w in enumerate(windows_order):
        y = df.loc[df["preproc"] == w, "delta_r2_mean"].to_numpy()
        jitter = rng.normal(0, 0.06, size=len(y))
        plt.scatter(np.full_like(y, x[i], dtype=float) + jitter, y, s=35)

    plt.axhline(0, linewidth=1)
    plt.xticks(x, windows_order)
    plt.ylabel("Δmax R²(best-PC) = classic_aug − baseline")
    plt.title("Effect Localization by Spectral Window (Best-PC per Factor)")
    plt.tight_layout()

    plt.savefig(OUTDIR / "fig1_dimdesc_windows.png", dpi=300)
    plt.savefig(OUTDIR / "fig1_dimdesc_windows.pdf")
    plt.close()


def fig2_pc_redistribution(
    path_sum: Path,
    window: str = "amide3",
    factor: str = "Anamnes_factor",
    use_mixup: int = 0,
    max_pc: int = 10,
) -> None:
    """
    Method figure: R² profiles across PCs (baseline vs classic_aug).
    Shows two vertical lines:
      - best ΔPC: argmax over (R²_aug - R²_base)
      - best baseline PC: argmax over R²_base
    This makes it clear why best-PC by Δ can differ from best absolute association.
    """
    df = pd.read_csv(path_sum)
    df = df[(df["preproc"] == window) & (df["factor"] == factor) & (df["use_mixup"] == use_mixup)].copy()

    if df.empty:
        raise ValueError(f"No rows found for window={window}, factor={factor}, use_mixup={use_mixup} in {path_sum}")

    # Parse PC number from strings like "PC9"
    df["pc_num"] = df["pc"].astype(str).str.extract(r"(\d+)").astype(int)
    df = df[df["pc_num"] <= max_pc].sort_values("pc_num")

    x = df["pc_num"].to_numpy()
    y_base = df["r2_baseline_mean"].to_numpy()
    y_aug = df["r2_aug_mean"].to_numpy()

    delta = y_aug - y_base
    best_delta_pc = int(x[np.argmax(delta)])
    best_abs_base_pc = int(x[np.argmax(y_base)])

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(x, y_base, marker="o", label="baseline")
    plt.plot(x, y_aug, marker="o", label="classic_aug")

    plt.axvline(best_delta_pc, linestyle="--", linewidth=1, label=f"best ΔPC = {best_delta_pc}")
    plt.axvline(best_abs_base_pc, linestyle=":", linewidth=1, label=f"best baseline PC = {best_abs_base_pc}")

    plt.xticks(x)
    plt.xlabel("PC")
    plt.ylabel("R² (factor ↔ PC score)")
    plt.title(f"{window}: {factor} (baseline vs classic_aug)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTDIR / f"fig2_pc_curve_{window}_{factor}.png", dpi=300)
    plt.savefig(OUTDIR / f"fig2_pc_curve_{window}_{factor}.pdf")
    plt.close()


def main() -> None:
    if not PATH_BEST.exists():
        raise FileNotFoundError(f"Not found: {PATH_BEST}")
    if not PATH_SUM.exists():
        raise FileNotFoundError(f"Not found: {PATH_SUM}")

    fig1_window_deltas(PATH_BEST, use_mixup=0)

    # Choose factor: Anamnes_factor / Age_factor / caries_factor / Gender / Parodont
    fig2_pc_redistribution(PATH_SUM, window="amide3", factor="Anamnes_factor", use_mixup=0, max_pc=10)

    print(f"Saved figures to: {OUTDIR}")


if __name__ == "__main__":
    main()