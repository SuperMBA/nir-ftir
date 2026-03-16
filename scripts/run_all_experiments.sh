#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MODULE="src.train_baselines"

DATA_COVID="${DATA_COVID:-data/processed/train.parquet}"
DATA_DIAB="${DATA_DIAB:-data/processed/diabetes_saliva.parquet}"

CLEAN_REPORTS="${CLEAN_REPORTS:-1}"   # 0/1
RUN_FRDA="${RUN_FRDA:-0}"             # 0/1

COVID_SEEDS_DEFAULT=(0 1 2)
DIAB_SEEDS_DEFAULT=(0 1 2)

if [[ -n "${COVID_SEEDS:-}" ]]; then read -r -a COVID_SEEDS <<< "${COVID_SEEDS}"; else COVID_SEEDS=("${COVID_SEEDS_DEFAULT[@]}"); fi
if [[ -n "${DIAB_SEEDS:-}" ]]; then read -r -a DIAB_SEEDS <<< "${DIAB_SEEDS}"; else DIAB_SEEDS=("${DIAB_SEEDS_DEFAULT[@]}"); fi

REPORT_DIR="reports/exp"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="${REPORT_DIR}/run_${RUN_TS}"

if [[ "${CLEAN_REPORTS}" == "1" ]]; then
  if [[ -d "${REPORT_DIR}" ]]; then mv "${REPORT_DIR}" "reports/exp_old_${RUN_TS}"; fi
fi
mkdir -p "${OUT_BASE}"

hr () { echo "------------------------------------------------------------"; }

run_one () {
  local out_subdir="$1"; shift
  "${PYTHON_BIN}" -m "${MODULE}" --outdir "${OUT_BASE}/${out_subdir}" "$@"
}

run_block_seeds () {
  local title="$1"; shift
  local out_subdir="$1"; shift
  local seeds_name="$1"; shift
  local -n SEEDS_REF="${seeds_name}"

  hr
  echo "${title} -> outdir=${OUT_BASE}/${out_subdir} (seeds=${#SEEDS_REF[@]})"
  for s in "${SEEDS_REF[@]}"; do
    echo "  -> seed=${s}"
    run_one "${out_subdir}" "$@" --seed "${s}"
  done
}

echo "=============================="
echo "A) COVID (train.parquet)"
echo "=============================="

PREPROC_COVID_D0=(--crop-min 800 --crop-max 1300 --sg-window 11 --sg-poly 2 --sg-deriv 0 --norm snv --xscale center)
PREPROC_COVID_D1=(--crop-min 800 --crop-max 1300 --sg-window 11 --sg-poly 2 --sg-deriv 1 --norm snv --xscale center)

COMMON_COVID=(
  --dataset covid_saliva
  --data-path "${DATA_COVID}"
  --group-col ID
  --protocol mcdcv_plsda
  --mc-iter 200
  --inner-splits 5
  --val-size 0.2
  --calib platt --calib-real-only --calib-frac 0.2
  --threshold-by recall_plus --recall-target 0.85 --min-spec 0.50
)

NO_AUG=(--search-aug fixed --noise-std 0 --noise-med 0 --shift 0 --scale 0 --tilt 0 --offset 0 --mixup 0 --mixwithin 0 --aug-repeats 1 --p-apply 0.5)
CLASSIC_AUG=(--search-aug fixed --noise-med 0.015 --shift 2.0 --mixup 0.4 --aug-repeats 1 --p-apply 0.5)
FRDA_AUG=(--frda-lite --frda-k 4 --frda-width 40 --frda-local-scale 0.02)

MODELS_COVID="${MODELS_COVID:-plsda,logreg,lda,svm_lin,svm_rbf}"

run_block_seeds "[A1] COVID baseline (no aug) D0" "A1_covid_baseline_d0" COVID_SEEDS \
  "${COMMON_COVID[@]}" "${PREPROC_COVID_D0[@]}" --models "${MODELS_COVID}" "${NO_AUG[@]}"

run_block_seeds "[A2] COVID classic aug D0" "A2_covid_classic_aug_d0" COVID_SEEDS \
  "${COMMON_COVID[@]}" "${PREPROC_COVID_D0[@]}" --models "${MODELS_COVID}" "${CLASSIC_AUG[@]}"

if [[ "${RUN_FRDA}" == "1" ]]; then
  run_block_seeds "[A3] COVID classic + FRDA-lite D0" "A3_covid_classic_frda_d0" COVID_SEEDS \
    "${COMMON_COVID[@]}" "${PREPROC_COVID_D0[@]}" --models "${MODELS_COVID}" "${CLASSIC_AUG[@]}" "${FRDA_AUG[@]}"
else
  hr; echo "[A3] FRDA-lite skipped (RUN_FRDA=0)"
fi

echo "=============================="
echo "B) DIABETES (diabetes_saliva.parquet)"
echo "=============================="

PREPROC_DIAB=(--crop-min 900 --crop-max 1800 --sg-window 11 --sg-poly 2 --sg-deriv 1 --norm snv --xscale center)

COMMON_DIAB=(
  --dataset diabetes_saliva
  --data-path "${DATA_DIAB}"
  --group-col sample_id
  --protocol cv_holdout
  --n-splits 5
  --val-size 0.2
  --meta-stratify age,gender --age-bins 5
  --calib platt --calib-real-only --calib-frac 0.3
  --threshold-by f1_plus --min-spec 0.65
)

DIAB_STRONG_AUG=(--search-aug fixed --mixwithin 0.4 --scale 0.02 --tilt 0.02 --offset 0.01 --aug-repeats 3 --p-apply 0.7)

MODELS_DIAB="${MODELS_DIAB:-plsda,logreg,lda,svm_rbf}"

run_block_seeds "[B1] DIAB baseline (no aug)" "B1_diab_baseline" DIAB_SEEDS \
  "${COMMON_DIAB[@]}" "${PREPROC_DIAB[@]}" --models "${MODELS_DIAB}" "${NO_AUG[@]}"

run_block_seeds "[B2] DIAB strong aug" "B2_diab_strong_aug" DIAB_SEEDS \
  "${COMMON_DIAB[@]}" "${PREPROC_DIAB[@]}" --models "${MODELS_DIAB}" "${DIAB_STRONG_AUG[@]}"

echo
echo "ALL DONE."
echo "Reports in: ${OUT_BASE}"
