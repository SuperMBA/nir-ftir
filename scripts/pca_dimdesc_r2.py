# scripts/pca_dimdesc_r2.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- make repo root importable (fixes ModuleNotFoundError: 'src') ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import train_baselines as tb  # noqa: E402


def _try_pvalue_anova(levels: list[np.ndarray]) -> float:
    """p-value via scipy if available; otherwise NaN."""
    try:
        from scipy.stats import f_oneway  # type: ignore

        return float(f_oneway(*levels).pvalue)
    except Exception:
        return float("nan")


def eta2_anova(x: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    """
    R² = eta² from one-way ANOVA: how much variance of x is explained by factor g.
    Returns (r2, p_value).
    """
    x = np.asarray(x, float)
    g = np.asarray(g)

    uniq = np.unique(g)
    levels = [x[g == lvl] for lvl in uniq]
    # keep only groups with >=2 points
    levels = [v for v in levels if len(v) >= 2]
    if len(levels) < 2:
        return np.nan, np.nan

    mu = float(x.mean())
    sst = float(((x - mu) ** 2).sum())
    sse = float(sum(((v - float(v.mean())) ** 2).sum() for v in levels))
    r2 = 1.0 - sse / sst if sst > 1e-12 else np.nan

    p = _try_pvalue_anova(levels)
    return float(r2), float(p)


def make_classic_aug(profile: str, use_mixup: bool) -> tb.AugConfig:
    # exactly your classic_mild defaults from run_gdb_study.sh (QC uses p_apply=1.0)
    if profile == "mild":
        return tb.AugConfig(
            search_aug="fixed",
            p_apply=1.0,
            noise_std=0.0,
            noise_med=0.004,
            shift=1.0,
            scale=0.004,
            tilt=0.003,
            offset=0.0015,
            mixup=0.08 if use_mixup else 0.0,
            mixwithin=0.0,
            aug_repeats=1,
        )
    # “full” just in case
    return tb.AugConfig(
        search_aug="fixed",
        p_apply=1.0,
        noise_std=0.0,
        noise_med=0.008,
        shift=2.0,
        scale=0.008,
        tilt=0.006,
        offset=0.003,
        mixup=0.15 if use_mixup else 0.0,
        mixwithin=0.0,
        aug_repeats=1,
    )


def preproc_params(profile: str):
    """
    Must match your run_gdb_qc_r2.sh profiles:
      - paper_full: 870-3400 drop 1800-2800 SG(25,2,0) SNV center
      - paper_low:  870-1800 drop none      SG(25,2,0) SNV center
      - amide3:     1185-1330 drop none     SG(25,2,0) SNV center
    """
    if profile == "paper_full":
        return dict(crop_min=870, crop_max=3400, drop="1800-2800", sg_w=25, sg_p=2, sg_d=0)
    if profile == "paper_low":
        return dict(crop_min=870, crop_max=1800, drop="", sg_w=25, sg_p=2, sg_d=0)
    if profile == "amide3":
        return dict(crop_min=1185, crop_max=1330, drop="", sg_w=25, sg_p=2, sg_d=0)
    raise ValueError(f"Unknown preproc-profile: {profile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--factors",
        required=True,
        help="Comma-separated factor columns, e.g. Gender,Age_factor,caries_factor,Parodont,Anamnes_factor",
    )
    ap.add_argument("--preproc-profile", required=True, choices=["paper_full", "paper_low", "amide3"])
    ap.add_argument("--pca-ncomp", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classic-profile", default="mild", choices=["mild", "full"])
    ap.add_argument("--use-mixup", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data_path)

    # spectral matrix
    spec_cols = tb.pick_spectral_columns(df)
    X, wn = tb.build_X_wn(df, spec_cols)

    pp = preproc_params(args.preproc_profile)
    drop_ranges = tb.parse_drop_ranges(pp["drop"])
    Xp, _ = tb.preprocess(
        X=X,
        wn=wn,
        crop_min=pp["crop_min"],
        crop_max=pp["crop_max"],
        sg_window=pp["sg_w"],
        sg_poly=pp["sg_p"],
        sg_deriv=pp["sg_d"],
        norm="snv",
        drop_ranges=drop_ranges,
    )

    # center (как у тебя в paper_full профиле)
    xsc = StandardScaler(with_mean=True, with_std=False)
    Xr = xsc.fit_transform(Xp).astype(np.float32)

    rng = np.random.default_rng(args.seed)

    # --- Baseline PCA (fit on real) ---
    ncomp = min(args.pca_ncomp, Xr.shape[0] - 1, Xr.shape[1])
    pca_base = PCA(n_components=ncomp, random_state=args.seed)
    Z_base_real = pca_base.fit_transform(Xr)

    # --- Augmented PCA: fit PCA on (real + classic_synth), evaluate only on real ---
    aug = make_classic_aug(args.classic_profile, use_mixup=bool(args.use_mixup))
    frda = tb.FrdaConfig(enabled=False, k=4, width=40, local_scale=0.02)

    # IMPORTANT: mixup uses y; for PCA we set dummy y but usually you should run use_mixup=0.
    y_dummy = np.zeros(len(Xr), dtype=int)
    Xa, _ = tb.build_augmented_train(Xr, y_dummy, aug=aug, rng=rng, frda=frda, frda_mask=None)

    pca_aug = PCA(n_components=ncomp, random_state=args.seed)
    pca_aug.fit(Xa)
    Z_aug_real = pca_aug.transform(Xr)

    factors = [c.strip() for c in args.factors.split(",") if c.strip()]

    rows = []
    for fac in factors:
        if fac not in df.columns:
            print(f"[WARN] factor '{fac}' not in df.columns -> skip")
            continue

        s = df[fac]

        # treat as qualitative (as in dimdesc) unless clearly numeric with many uniq
        is_num = pd.api.types.is_numeric_dtype(s)
        nunique = int(pd.Series(s).nunique(dropna=True))
        fac_type = "quant" if (is_num and nunique > 10) else "qual"

        if fac_type == "qual":
            z = s.astype(str).fillna("NA").str.strip().to_numpy()  # strip is CRITICAL for Parodont: "G "
        else:
            z = pd.to_numeric(s, errors="coerce").to_numpy()

        for k in range(ncomp):
            pc = f"PC{k+1}"
            xb = Z_base_real[:, k]
            xa = Z_aug_real[:, k]

            if fac_type == "qual":
                r2b, pb = eta2_anova(xb, z)
                r2a, pa = eta2_anova(xa, z)
            else:
                # quick r^2 for quantitative
                m = np.isfinite(z)
                if m.sum() < 3:
                    r2b, pb, r2a, pa = np.nan, np.nan, np.nan, np.nan
                else:
                    zb = z[m]
                    rb = np.corrcoef(xb[m], zb)[0, 1]
                    ra = np.corrcoef(xa[m], zb)[0, 1]
                    r2b, r2a = float(rb * rb), float(ra * ra)
                    pb, pa = np.nan, np.nan

            rows.append(
                dict(
                    factor=fac,
                    type=fac_type,
                    pc=pc,
                    r2_baseline=r2b,
                    p_baseline=pb,
                    r2_aug=r2a,
                    p_aug=pa,
                    delta_r2=(r2a - r2b) if np.isfinite(r2a) and np.isfinite(r2b) else np.nan,
                    preproc=args.preproc_profile,
                    classic_profile=args.classic_profile,
                    use_mixup=int(args.use_mixup),
                    seed=int(args.seed),
                )
            )

    out = pd.DataFrame(rows).sort_values(["factor", "pc"])
    out.to_csv(outdir / "dimdesc_like_r2.csv", index=False)
    print(f"[OK] saved: {outdir/'dimdesc_like_r2.csv'} rows={len(out)}")


if __name__ == "__main__":
    main()