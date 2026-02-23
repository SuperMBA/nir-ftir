#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Korea experiments (balanced: quality vs time)
# Main focus: D1 + strong augmentation + SVM RBF included
# ------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODULE="src.train_baselines"

# parquet (as in your workflow)
DATA_KOREA="${DATA_KOREA:-data/processed/korea_main_FINAL_bestrep_train.parquet}"

# housekeeping
CLEAN_REPORTS="${CLEAN_REPORTS:-1}"   # 0/1
RUN_FRDA="${RUN_FRDA:-0}"             # 0/1  (expensive)
RUN_D0_ABLATION="${RUN_D0_ABLATION:-0}" # 0/1 (optional derivative ablation)

# speed/quality knobs
MC_ITER="${MC_ITER:-120}"             # balanced default (raise to 200 for final)
INNER_SPLITS="${INNER_SPLITS:-5}"
VAL_SIZE="${VAL_SIZE:-0.2}"

# seeds
KOREA_SEEDS_DEFAULT=(0 1 2)
if [[ -n "${KOREA_SEEDS:-}" ]]; then
  read -r -a KOREA_SEEDS <<< "${KOREA_SEEDS}"
else
  KOREA_SEEDS=("${KOREA_SEEDS_DEFAULT[@]}")
fi

# optional smaller seed list for FRDA (to save time)
FRDA_SEEDS_DEFAULT=(0 1)
if [[ -n "${FRDA_SEEDS:-}" ]]; then
  read -r -a FRDA_SEEDS <<< "${FRDA_SEEDS}"
else
  FRDA_SEEDS=("${FRDA_SEEDS_DEFAULT[@]}")
fi

REPORT_DIR="reports/exp"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="${REPORT_DIR}/korea_run_${RUN_TS}"

if [[ ! -f "${DATA_KOREA}" ]]; then
  echo "[ERR] parquet not found: ${DATA_KOREA}"
  exit 1
fi

if [[ "${CLEAN_REPORTS}" == "1" ]]; then
  if [[ -d "${REPORT_DIR}" ]]; then
    mv "${REPORT_DIR}" "reports/exp_old_${RUN_TS}"
  fi
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
  local -n SEEDS_REF="$1"; shift

  hr
  echo "${title} -> outdir=${OUT_BASE}/${out_subdir}"
  echo "seeds=(${SEEDS_REF[*]})"
  for s in "${SEEDS_REF[@]}"; do
    echo "  -> seed=${s}"
    run_one "${out_subdir}" "$@" --seed "${s}"
  done
}

echo "=============================="
echo "KOREA (bestrep) -> ${DATA_KOREA}"
echo "MC_ITER=${MC_ITER}, seeds=${KOREA_SEEDS[*]}"
echo "=============================="

# ------------------------------------------------------------
# Preprocessing
# D1 = main (better signal for subtle FTIR differences)
# D0 = optional ablation only
# ------------------------------------------------------------
PREPROC_D1=(--crop-min 900 --crop-max 1800 --sg-window 11 --sg-poly 2 --sg-deriv 1 --norm snv --xscale center)
PREPROC_D0=(--crop-min 900 --crop-max 1800 --sg-window 11 --sg-poly 2 --sg-deriv 0 --norm snv --xscale center)

# ------------------------------------------------------------
# Common protocol (small N -> mcdcv_plsda)
# ------------------------------------------------------------
COMMON_KOREA=(
  --dataset diabetes_saliva
  --data-path "${DATA_KOREA}"
  --label-col y
  --group-col ID
  --protocol mcdcv_plsda
  --mc-iter "${MC_ITER}"
  --inner-splits "${INNER_SPLITS}"
  --val-size "${VAL_SIZE}"
  --calib platt --calib-real-only --calib-frac 0.2
  --threshold-by f1_plus --min-spec 0.65
  --tag korea
)

# ------------------------------------------------------------
# Augmentation presets
# ------------------------------------------------------------
NO_AUG=(
  --search-aug fixed
  --noise-std 0 --noise-med 0 --shift 0
  --scale 0 --tilt 0 --offset 0
  --mixup 0 --mixwithin 0
  --aug-repeats 1 --p-apply 0.5
)

# "Strong" preset (inspired by diabetes script, usually better for D1)
STRONG_AUG=(
  --search-aug fixed
  --mixwithin 0.4
  --scale 0.02
  --tilt 0.02
  --offset 0.01
  --aug-repeats 3
  --p-apply 0.7
)

# Optional second classic preset (noise+shift+mixup) for comparison
CLASSIC_AUG=(
  --search-aug fixed
  --noise-med 0.010
  --shift 1.0
  --mixup 0.3
  --mixwithin 0.2
  --aug-repeats 1
  --p-apply 0.5
)

FRDA_AUG=(--frda-lite --frda-k 4 --frda-width 40 --frda-local-scale 0.02)

# ------------------------------------------------------------
# Models
# Main set includes SVM RBF
# Optional targeted subsets for expensive runs
# ------------------------------------------------------------
MODELS_MAIN="${MODELS_MAIN:-plsda,logreg,lda,svm_lin,svm_rbf}"
MODELS_FAST_ABL="${MODELS_FAST_ABL:-plsda,lda,svm_rbf}"
MODELS_FRDA="${MODELS_FRDA:-plsda,svm_rbf}"

# ------------------------------------------------------------
# Stage 1 (main results): D1 baseline vs strong_aug
# ------------------------------------------------------------
run_block_seeds "[K1] Korea baseline D1 (main)" "K1_korea_baseline_d1" KOREA_SEEDS \
  "${COMMON_KOREA[@]}" "${PREPROC_D1[@]}" --models "${MODELS_MAIN}" "${NO_AUG[@]}"

run_block_seeds "[K2] Korea strong_aug D1 (main)" "K2_korea_strong_aug_d1" KOREA_SEEDS \
  "${COMMON_KOREA[@]}" "${PREPROC_D1[@]}" --models "${MODELS_MAIN}" "${STRONG_AUG[@]}"

# ------------------------------------------------------------
# Stage 2 (optional): compare another classic aug preset
# Useful if you want to show augmentation-type effect, but skip if time is tight
# ------------------------------------------------------------
if [[ "${RUN_CLASSIC_COMPARE:-1}" == "1" ]]; then
  run_block_seeds "[K3] Korea classic_aug D1 (compare)" "K3_korea_classic_aug_d1" KOREA_SEEDS \
    "${COMMON_KOREA[@]}" "${PREPROC_D1[@]}" --models "${MODELS_MAIN}" "${CLASSIC_AUG[@]}"
else
  hr; echo "[K3] classic compare skipped (RUN_CLASSIC_COMPARE=0)"
fi

# ------------------------------------------------------------
# Stage 3 (optional): D0 ablation only on a reduced model set
# ------------------------------------------------------------
if [[ "${RUN_D0_ABLATION}" == "1" ]]; then
  run_block_seeds "[K4] Korea baseline D0 (ablation)" "K4_korea_baseline_d0_ablation" KOREA_SEEDS \
    "${COMMON_KOREA[@]}" "${PREPROC_D0[@]}" --models "${MODELS_FAST_ABL}" "${NO_AUG[@]}"

  run_block_seeds "[K5] Korea strong_aug D0 (ablation)" "K5_korea_strong_aug_d0_ablation" KOREA_SEEDS \
    "${COMMON_KOREA[@]}" "${PREPROC_D0[@]}" --models "${MODELS_FAST_ABL}" "${STRONG_AUG[@]}"
else
  hr; echo "[K4/K5] D0 ablation skipped (RUN_D0_ABLATION=0)"
fi

# ------------------------------------------------------------
# Stage 4 (optional): FRDA-lite only on top candidate models, fewer seeds
# ------------------------------------------------------------
if [[ "${RUN_FRDA}" == "1" ]]; then
  run_block_seeds "[K6] Korea strong_aug + FRDA D1 (targeted)" "K6_korea_strong_frda_d1" FRDA_SEEDS \
    "${COMMON_KOREA[@]}" "${PREPROC_D1[@]}" --models "${MODELS_FRDA}" "${STRONG_AUG[@]}" "${FRDA_AUG[@]}"
else
  hr; echo "[K6] FRDA-lite skipped (RUN_FRDA=0)"
fi

echo
echo "ALL DONE."
echo "Reports in: ${OUT_BASE}"
echo "Next: python scripts/aggregate_reports_korea.py"
echo
echo "Tip (final quality run): rerun best setup with MC_ITER=200"
