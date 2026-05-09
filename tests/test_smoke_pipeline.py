from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _make_toy_ftir_parquet(path: Path, n: int = 48, p: int = 80) -> None:
    """Create a tiny FTIR-like binary dataset for a fast end-to-end smoke test.

    The test is intentionally synthetic: it checks that the pipeline can load a
    parquet file, detect numeric spectral columns, preprocess spectra, train a
    model, write a JSON report, and keep train-only augmentation callable.
    It does NOT validate scientific quality.
    """
    rng = np.random.default_rng(42)

    wn = np.linspace(800, 1300, p)
    y = np.array([0, 1] * (n // 2), dtype=int)

    # Smooth FTIR-like baseline + a small class-dependent band.
    baseline = (
        0.4 * np.sin(wn / 80.0)[None, :]
        + 0.2 * np.cos(wn / 55.0)[None, :]
    )
    class_band = np.exp(-0.5 * ((wn - 1070.0) / 25.0) ** 2)[None, :]
    X = baseline + y[:, None] * 0.35 * class_band + rng.normal(0, 0.04, size=(n, p))

    df = pd.DataFrame(X, columns=[f"{v:.1f}" for v in wn])
    df.insert(0, "ID", [f"S{i:03d}" for i in range(n)])
    df.insert(1, "y", y)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_train_baselines_toy_baseline_and_augmented(tmp_path: Path) -> None:
    data_path = tmp_path / "toy_ftir.parquet"
    out_base = tmp_path / "baseline"
    out_aug = tmp_path / "augmented"
    _make_toy_ftir_parquet(data_path)

    common = [
        sys.executable,
        "-m",
        "src.train_baselines",
        "--dataset",
        "covid_saliva",
        "--data-path",
        str(data_path),
        "--label-col",
        "y",
        "--group-col",
        "ID",
        "--protocol",
        "cv_holdout",
        "--models",
        "logreg",
        "--seed",
        "0",
        "--n-splits",
        "2",
        "--inner-splits",
        "2",
        "--val-size",
        "0.25",
        "--crop-min",
        "800",
        "--crop-max",
        "1300",
        "--sg-window",
        "5",
        "--sg-poly",
        "2",
        "--sg-deriv",
        "0",
        "--norm",
        "snv",
        "--xscale",
        "center",
        "--calib",
        "none",
        "--threshold-by",
        "none",
        "--p-apply",
        "0.5",
        "--aug-repeats",
        "1",
    ]

    baseline_cmd = common + [
        "--noise-std", "0",
        "--noise-med", "0",
        "--shift", "0",
        "--scale", "0",
        "--tilt", "0",
        "--offset", "0",
        "--mixup", "0",
        "--mixwithin", "0",
        "--tag", "smoke_baseline",
        "--outdir", str(out_base),
    ]

    augmented_cmd = common + [
        "--noise-med", "0.003",
        "--shift", "1.0",
        "--mixup", "0.05",
        "--tag", "smoke_augmented",
        "--outdir", str(out_aug),
    ]

    subprocess.run(baseline_cmd, cwd=ROOT, check=True)
    subprocess.run(augmented_cmd, cwd=ROOT, check=True)

    json_files = sorted(out_base.glob("*.json")) + sorted(out_aug.glob("*.json"))
    assert len(json_files) == 2

    for path in json_files:
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["protocol"] == "cv_holdout"
        assert "results" in report
        assert "logreg" in report["results"]
        test_metrics = report["results"]["logreg"]["test"]
        for metric in ["f1", "recall", "spec", "pr_auc", "brier", "ece"]:
            assert metric in test_metrics
