#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-.}"

DATA_GDB="${DATA_GDB:-data/processed/gdb_smalln.parquet}"
OUT_ROOT="${OUT_ROOT:-reports/qc_r2}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_ROOT}/gdb_qc_r2_${RUN_TS}"
mkdir -p "${OUT_DIR}"

# smoke/main
MODE="${MODE:-main}"   # smoke | main
if [[ "${MODE}" == "smoke" ]]; then
  SEEDS="${SEEDS:-0}"
  CV_SPLITS="${CV_SPLITS:-3}"
  CV_REPEATS="${CV_REPEATS:-2}"
  VAE_EPOCHS="${VAE_EPOCHS:-120}"
  WGAN_STEPS="${WGAN_STEPS:-300}"
else
  SEEDS="${SEEDS:-0,1,2,3,4}"
  CV_SPLITS="${CV_SPLITS:-3}"
  CV_REPEATS="${CV_REPEATS:-10}"
  VAE_EPOCHS="${VAE_EPOCHS:-400}"
  WGAN_STEPS="${WGAN_STEPS:-1200}"
fi

# allow "0 1 2" -> "0,1,2"
SEEDS="$(printf "%s" "${SEEDS}" | tr -s '[:space:]' ',' | sed 's/^,*//; s/,*$//')"

# Supervisor binary tasks
LABELS="${LABELS:-y_parodont_H_vs_path,y_healthy_vs_any,y_anamnes_H_vs_path}"

# baseline/classic/vae/wgan
METHODS="${METHODS:-baseline,classic,vae,wgan}"
CLASSIC_PROFILE="${CLASSIC_PROFILE:-mild}"   # mild|full (passed to generator)

# Preproc profiles aligned with papers
PREPROC_PROFILE="${PREPROC_PROFILE:-paper_full}"  # paper_full|paper_low|legacy|amide3

# Defaults = paper_full (6395: 3400–2800 + 1800–870; SG(25), deriv=0)
CROP_MIN="${CROP_MIN:-870}"
CROP_MAX="${CROP_MAX:-3400}"
DROP_RANGES="${DROP_RANGES:-1800-2800}"
SG_WINDOW="${SG_WINDOW:-25}"
SG_POLY="${SG_POLY:-2}"
SG_DERIV="${SG_DERIV:-0}"
NORM="${NORM:-snv}"
XSCALE="${XSCALE:-center}"

if [[ "${PREPROC_PROFILE}" == "paper_low" ]]; then
  CROP_MIN=870; CROP_MAX=1800; DROP_RANGES=""
  SG_WINDOW=25; SG_POLY=2; SG_DERIV=0
elif [[ "${PREPROC_PROFILE}" == "legacy" ]]; then
  CROP_MIN=800; CROP_MAX=1800; DROP_RANGES=""
  SG_WINDOW=11; SG_POLY=2; SG_DERIV=1
elif [[ "${PREPROC_PROFILE}" == "amide3" ]]; then
  # from 4693: Amide III ~ 1330–1185 cm^-1
  CROP_MIN=1185; CROP_MAX=1330; DROP_RANGES=""
  SG_WINDOW=25; SG_POLY=2; SG_DERIV=0
fi

# Generators (conservative)
N_SYNTH_MULT="${N_SYNTH_MULT:-1.0}"
GEN_PCA_VAR="${GEN_PCA_VAR:-0.95}"
GEN_PCA_MAX="${GEN_PCA_MAX:-8}"

# PLS (R2/Q2 block)
PLS_NCOMP="${PLS_NCOMP:-2}"

# QC
KNN_K="${KNN_K:-5}"
DEVICE="${DEVICE:-auto}"  # auto/cpu/cuda

echo "====================================="
echo "GDB small-n | AE/WGAN + QC + PLS R2/Q2"
echo "Data:   ${DATA_GDB}"
echo "Out:    ${OUT_DIR}"
echo "Mode:   ${MODE}"
echo "Seeds:  ${SEEDS}"
echo "CV:     splits=${CV_SPLITS}, repeats=${CV_REPEATS}"
echo "Labels: ${LABELS}"
echo "Methods:${METHODS} (classic_profile=${CLASSIC_PROFILE})"
echo "Preproc:${PREPROC_PROFILE} crop=${CROP_MIN}-${CROP_MAX} drop='${DROP_RANGES}' SG(${SG_WINDOW},p=${SG_POLY},d=${SG_DERIV})"
echo "====================================="

"${PYTHON_BIN}" scripts/gdb_qc_r2_generators.py \
  --data-path "${DATA_GDB}" \
  --outdir "${OUT_DIR}" \
  --label-cols "${LABELS}" \
  --methods "${METHODS}" \
  --classic-profile "${CLASSIC_PROFILE}" \
  --group-col sample_id \
  --crop-min "${CROP_MIN}" \
  --crop-max "${CROP_MAX}" \
  --drop-ranges "${DROP_RANGES}" \
  --sg-window "${SG_WINDOW}" \
  --sg-poly "${SG_POLY}" \
  --sg-deriv "${SG_DERIV}" \
  --norm "${NORM}" \
  --xscale "${XSCALE}" \
  --seeds "${SEEDS}" \
  --cv-splits "${CV_SPLITS}" \
  --cv-repeats "${CV_REPEATS}" \
  --n-synth-mult "${N_SYNTH_MULT}" \
  --gen-pca-var "${GEN_PCA_VAR}" \
  --gen-pca-max "${GEN_PCA_MAX}" \
  --vae-epochs "${VAE_EPOCHS}" \
  --wgan-steps "${WGAN_STEPS}" \
  --pls-ncomp "${PLS_NCOMP}" \
  --knn-k "${KNN_K}" \
  --device "${DEVICE}"

echo
echo "[DONE] Outputs:"
echo "  ${OUT_DIR}/per_fold_metrics.csv"
echo "  ${OUT_DIR}/summary_label_method.csv"
echo "  ${OUT_DIR}/summary_method.csv"
echo "  ${OUT_DIR}/config.json"