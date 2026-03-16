#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# GDB small-n supervised stability experiments (binary only)
# ------------------------------------------------------------
# Uses existing src.train_baselines pipeline.
# Focus: stability/variance across seeds and augmentation as a factor.
#
# SPEED / REPRO goals:
# - parallelize across seeds (N_JOBS)
# - skip seeds already computed (SKIP_EXISTING)
# - optional PLS-DA tuning (PLS_TUNE) with protocol mcdcv_plsda
# ============================================================

PYTHON_BIN="${PYTHON_BIN:-python}"
MODULE="${MODULE:-src.train_baselines}"
export PYTHONPATH="${PYTHONPATH:-.}"

DATA_GDB="${DATA_GDB:-data/processed/gdb_smalln.parquet}"

# ------------- SPEED switches -------------
# Parallel jobs (seeds in parallel). For 8 CPUs: start with 4.
N_JOBS="${N_JOBS:-4}"
# Skip seeds where a *.json already exists in seed folder
SKIP_EXISTING="${SKIP_EXISTING:-1}"
# Fail-soft mode (0=stop on first failure, 1=continue and log failures)
SOFT_FAIL="${SOFT_FAIL:-0}"

# Recommend: prevent BLAS oversubscription when running N_JOBS>1
# export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# ------------------------------
# Seeds
# ------------------------------
SEEDS_DEFAULT=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
if [[ -n "${GDB_SEEDS:-}" ]]; then
  _SEEDS_NORM="$(printf "%s" "${GDB_SEEDS}" | tr -d '\r' | tr -s '[:space:]' ' ' | sed 's/^ //; s/ $//')"
  IFS=' ' read -r -a GDB_SEEDS_ARR <<< "${_SEEDS_NORM}"
else
  GDB_SEEDS_ARR=("${SEEDS_DEFAULT[@]}")
fi

# ------------------------------
# Tasks (binary only)
# ------------------------------
RUN_PARODONT="${RUN_PARODONT:-1}"       # y_parodont_H_vs_path
RUN_ANAMNES="${RUN_ANAMNES:-1}"         # y_anamnes_H_vs_path
RUN_HEALTHY_ANY="${RUN_HEALTHY_ANY:-1}" # y_healthy_vs_any (stress-like)
# Appendix tasks (off by default)
RUN_CARIES_BIN="${RUN_CARIES_BIN:-0}"       # y_caries_C_vs_nonC
RUN_CARIES_H_PATH="${RUN_CARIES_H_PATH:-0}" # y_caries_H_vs_path (stress-like)

# ------------------------------
# Variants (augmentation)
# ------------------------------
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_CLASSIC_MILD="${RUN_CLASSIC_MILD:-1}"
RUN_CLASSIC_FULL="${RUN_CLASSIC_FULL:-0}"
RUN_FRDA_LITE="${RUN_FRDA_LITE:-0}"

# ------------------------------
# Protocols
# ------------------------------
PROTOCOL_MAIN="${PROTOCOL_MAIN:-mcdcv}"      # mcdcv | mcdcv_plsda
PROTOCOL_STRESS="${PROTOCOL_STRESS:-mcdcv}"  # mcdcv
if [[ "${PROTOCOL_MAIN}" == "loocv" || "${PROTOCOL_STRESS}" == "loocv" ]]; then
  echo "[ERROR] LOOCV requested, but ${MODULE} supports only: cv_holdout, mcdcv, mcdcv_plsda"
  exit 2
fi

# ------------------------------
# CV knobs
# ------------------------------
MC_ITER_MAIN="${MC_ITER_MAIN:-80}"
OUTER_FRAC_MAIN="${OUTER_FRAC_MAIN:-0.25}"
INNER_SPLITS_MAIN="${INNER_SPLITS_MAIN:-4}"
PLS_INNER_SPLITS_MAIN="${PLS_INNER_SPLITS_MAIN:-3}"
CALIB_MAIN="${CALIB_MAIN:-platt}"
CALIB_FRAC_MAIN="${CALIB_FRAC_MAIN:-0.25}"
THRESH_MAIN_MODE="${THRESH_MAIN_MODE:-f1_plus}"
MIN_SPEC_MAIN="${MIN_SPEC_MAIN:-0.50}"
RECALL_TARGET_MAIN="${RECALL_TARGET_MAIN:-0.80}"

MC_ITER_STRESS="${MC_ITER_STRESS:-60}"
OUTER_FRAC_STRESS="${OUTER_FRAC_STRESS:-0.22}"
INNER_SPLITS_STRESS="${INNER_SPLITS_STRESS:-2}"
PLS_INNER_SPLITS_STRESS="${PLS_INNER_SPLITS_STRESS:-2}"
THRESH_STRESS_MODE="${THRESH_STRESS_MODE:-recall_plus}"
RECALL_TARGET_STRESS="${RECALL_TARGET_STRESS:-0.80}"
MIN_SPEC_STRESS="${MIN_SPEC_STRESS:-0.30}"

# ------------------------------
# Meta-stratification (optional)
# ------------------------------
RUN_META="${RUN_META:-0}"
ALLOW_META_FAIL="${ALLOW_META_FAIL:-1}"
META_STRAT_COLS="${META_STRAT_COLS:-Gender}"  # e.g. "Gender" or "Age_factor,Gender"

# ------------------------------
# Preprocessing profile
#   auto  : choose paper_full if wn_max>=3300 else paper_low else legacy
#   paper : force paper_full if possible, else paper_low
#   legacy: old (800-1800, SG11, deriv=1)
# ------------------------------
PREPROC_PROFILE="${PREPROC_PROFILE:-auto}"

# ------------------------------
# Misc
# ------------------------------
SUPPRESS_SK_WARNINGS="${SUPPRESS_SK_WARNINGS:-1}"
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"

REPORTS_ROOT="${REPORTS_ROOT:-reports/exp}"
REPORTS_SUMMARY_DIR="${REPORTS_SUMMARY_DIR:-reports}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-gdb_smalln_${RUN_TS}}"
OUT_BASE="${OUT_BASE:-${REPORTS_ROOT}/${RUN_NAME}}"
mkdir -p "${OUT_BASE}"

FAILED_RUNS=()
FAILED_BLOCKS=()

trap 'echo "[ERROR] Script failed at line $LINENO: $BASH_COMMAND" >&2' ERR
hr () { echo "------------------------------------------------------------"; }

# ------------------------------------------------------------
# Preflight
# ------------------------------------------------------------
if [[ ! -f "${DATA_GDB}" ]]; then
  echo "[ERROR] Data file not found: ${DATA_GDB}" >&2
  exit 1
fi

# ------------------------------------------------------------
# Detect spectral min/max safely (never fail script)
# ------------------------------------------------------------
WN_LINE=$(
  (
    DATA_GDB="${DATA_GDB}" "${PYTHON_BIN}" - <<'PY'
import os
import pandas as pd
try:
    import src.train_baselines as tb
except Exception:
    import train_baselines as tb

p = os.environ["DATA_GDB"]
df = pd.read_parquet(p)

spec_cols = tb.pick_spectral_columns(df)
vals = []
for c in spec_cols:
    try:
        vals.append(float(c))
    except Exception:
        pass

if not vals:
    for c in df.columns:
        if isinstance(c, (int, float)):
            vals.append(float(c))

if not vals:
    print("nan nan")
else:
    print(f"{min(vals)} {max(vals)}")
PY
  ) 2>/dev/null || echo "nan nan"
)
IFS=' ' read -r WN_MIN WN_MAX <<< "${WN_LINE}"

# ------------------------------------------------------------
# Decide preprocessing based on profile + availability
# ------------------------------------------------------------
PREPROC_NAME="legacy"
CROP_MIN=800
CROP_MAX=1800
DROP_RANGES=""
SG_WINDOW=11
SG_POLY=2
SG_DERIV=1
NORM="snv"
XSCALE="center"

# paper-like per MDPI 6395: two zones 3400–2800 and 1800–870, SG(25), deriv=0
# (ALS baseline correction not reproduced in this pipeline)
if [[ "${PREPROC_PROFILE}" == "auto" || "${PREPROC_PROFILE}" == "paper" ]]; then
  awk_check=$(awk -v mx="${WN_MAX}" 'BEGIN{
    if(mx ~ /nan/i) { print "legacy"; exit }
    if(mx>=3300) print "full";
    else if(mx>=1750) print "low";
    else print "legacy";
  }')

  if [[ "${PREPROC_PROFILE}" == "paper" && "${awk_check}" == "legacy" ]]; then
    echo "[WARN] PREPROC_PROFILE=paper but spectrum max < 1750. Falling back to legacy." >&2
    awk_check="legacy"
  fi

  if [[ "${awk_check}" == "full" ]]; then
    PREPROC_NAME="paper_full"
    CROP_MIN=870
    CROP_MAX=3400
    DROP_RANGES="1800-2800"
    SG_WINDOW=25
    SG_POLY=2
    SG_DERIV=0
  elif [[ "${awk_check}" == "low" ]]; then
    PREPROC_NAME="paper_low"
    CROP_MIN=870
    CROP_MAX=1800
    DROP_RANGES=""
    SG_WINDOW=25
    SG_POLY=2
    SG_DERIV=0
  else
    PREPROC_NAME="legacy"
  fi
fi

if [[ "${PREPROC_PROFILE}" == "legacy" ]]; then
  PREPROC_NAME="legacy"
fi

PREPROC_GDB=(
  --crop-min "${CROP_MIN}" --crop-max "${CROP_MAX}"
  --sg-window "${SG_WINDOW}" --sg-poly "${SG_POLY}" --sg-deriv "${SG_DERIV}"
  --norm "${NORM}"
  --xscale "${XSCALE}"
)
if [[ -n "${DROP_RANGES}" ]]; then
  PREPROC_GDB+=( --drop-ranges "${DROP_RANGES}" )
fi

# ------------------------------
# Models + Optional PLS tuning
# ------------------------------
MODELS="${MODELS:-plsda,logreg,lda,svm_lin}"

COMMON_BASE=(
  --dataset diabetes_saliva
  --data-path "${DATA_GDB}"
  --group-col sample_id
  --models "${MODELS}"
)

# --- Optional PLS tuning (only affects PLS-DA) ---
PLS_TUNE="${PLS_TUNE:-0}"
PLS_GRID="${PLS_GRID:-2,3,4,5,6}"
if [[ "${PLS_TUNE}" == "1" ]]; then
  COMMON_BASE+=( --pls-tune --pls-grid "${PLS_GRID}" )
fi

# ------------------------------
# Augmentations
# ------------------------------
NO_AUG=(
  --search-aug fixed
  --noise-std 0 --noise-med 0
  --shift 0 --scale 0 --tilt 0 --offset 0
  --mixup 0 --mixwithin 0
  --aug-repeats 1
  --p-apply 0.50
)

CLASSIC_MILD_AUG=(
  --search-aug fixed
  --noise-med 0.004
  --shift 1.0
  --scale 0.004
  --tilt 0.003
  --offset 0.0015
  --mixup 0.08
  --mixwithin 0.00
  --aug-repeats 1
  --p-apply 0.30
)

CLASSIC_FULL_AUG=(
  --search-aug fixed
  --noise-med 0.008
  --shift 2.0
  --scale 0.008
  --tilt 0.006
  --offset 0.003
  --mixup 0.15
  --mixwithin 0.00
  --aug-repeats 1
  --p-apply 0.40
)

FRDA_LITE_AUG=(
  --search-aug fixed
  --noise-std 0 --noise-med 0
  --shift 0 --scale 0 --tilt 0 --offset 0
  --mixup 0 --mixwithin 0
  --aug-repeats 1
  --p-apply 0.35
  --frda-lite
  --frda-k 4
  --frda-width 40
  --frda-local-scale 0.015
)

META_STRAT=( --meta-stratify "${META_STRAT_COLS}" )

# ------------------------------------------------------------
# Runner helpers
# ------------------------------------------------------------
run_one () {
  local out_subdir="$1"; shift
  mkdir -p "${OUT_BASE}/${out_subdir}"
  local tag="${out_subdir//\//_}"

  if [[ "${SUPPRESS_SK_WARNINGS}" == "1" ]]; then
    PYTHONWARNINGS="ignore::FutureWarning" \
      "${PYTHON_BIN}" -m "${MODULE}" --outdir "${OUT_BASE}/${out_subdir}" --tag "${tag}" "$@"
  else
    "${PYTHON_BIN}" -m "${MODULE}" --outdir "${OUT_BASE}/${out_subdir}" --tag "${tag}" "$@"
  fi
}

# ---------- PARALLEL + SKIP ----------
run_block_seeds () {
  local title="$1"; shift
  local out_subdir="$1"; shift

  hr
  echo "${title} -> outdir=${OUT_BASE}/${out_subdir} (seeds=${#GDB_SEEDS_ARR[@]}) [N_JOBS=${N_JOBS}, SKIP_EXISTING=${SKIP_EXISTING}]"

  local running=0
  for s in "${GDB_SEEDS_ARR[@]}"; do
    local seed_dir="${OUT_BASE}/${out_subdir}/seed_${s}"
    mkdir -p "${seed_dir}"

    if [[ "${SKIP_EXISTING}" == "1" ]] && compgen -G "${seed_dir}/*.json" > /dev/null; then
      echo "  -> seed=${s} [skip existing]"
      continue
    fi

    echo "  -> seed=${s}"

    (
      if [[ "${SOFT_FAIL}" == "1" ]]; then
        if ! run_one "${out_subdir}/seed_${s}" "$@" --seed "${s}"; then
          echo "  [WARN] failed: ${out_subdir}/seed_${s}" >&2
        fi
      else
        run_one "${out_subdir}/seed_${s}" "$@" --seed "${s}"
      fi
    ) &

    ((running+=1))
    if (( running >= N_JOBS )); then
      wait -n
      ((running-=1))
    fi
  done
  wait
}

# Soft meta block (kept sequential; meta can be fragile, easier to read failures)
run_block_seeds_soft () {
  local title="$1"; shift
  local out_subdir="$1"; shift

  hr
  echo "${title} -> outdir=${OUT_BASE}/${out_subdir} (seeds=${#GDB_SEEDS_ARR[@]}) [soft mode]"
  local had_fail=0
  for s in "${GDB_SEEDS_ARR[@]}"; do
    echo "  -> seed=${s}"
    if run_one "${out_subdir}/seed_${s}" "$@" --seed "${s}"; then
      :
    else
      had_fail=1
      echo "  [WARN] failed seed=${s} in ${out_subdir} (continuing)"
    fi
  done
  if [[ "${had_fail}" == "1" ]]; then
    FAILED_BLOCKS+=("${out_subdir}")
  fi
}

run_block_meta () {
  if [[ "${ALLOW_META_FAIL}" == "1" ]]; then
    run_block_seeds_soft "$@"
  else
    run_block_seeds "$@"
  fi
}

run_task_family () {
  local slug="$1"
  local label_col="$2"
  local profile="$3"
  local human_title="$4"

  local -a cv_args=()
  local -a thr_args=()

  case "${profile}" in
    main)
      cv_args=(
        --protocol "${PROTOCOL_MAIN}"
        --mc-iter "${MC_ITER_MAIN}"
        --val-size "${OUTER_FRAC_MAIN}"
        --n-splits "${INNER_SPLITS_MAIN}"
        --inner-splits "${PLS_INNER_SPLITS_MAIN}"
        --calib "${CALIB_MAIN}"
        --calib-real-only
        --calib-frac "${CALIB_FRAC_MAIN}"
      )
      case "${THRESH_MAIN_MODE}" in
        none)        thr_args=( --threshold-by none ) ;;
        f1_plus)     thr_args=( --threshold-by f1_plus --min-spec "${MIN_SPEC_MAIN}" ) ;;
        recall_plus) thr_args=( --threshold-by recall_plus --recall-target "${RECALL_TARGET_MAIN}" --min-spec "${MIN_SPEC_MAIN}" ) ;;
        *) echo "[ERROR] Unsupported THRESH_MAIN_MODE=${THRESH_MAIN_MODE}"; exit 2 ;;
      esac
      ;;
    stress)
      cv_args=(
        --protocol "${PROTOCOL_STRESS}"
        --mc-iter "${MC_ITER_STRESS}"
        --val-size "${OUTER_FRAC_STRESS}"
        --n-splits "${INNER_SPLITS_STRESS}"
        --inner-splits "${PLS_INNER_SPLITS_STRESS}"
        --calib none
      )
      case "${THRESH_STRESS_MODE}" in
        none)        thr_args=( --threshold-by none ) ;;
        f1_plus)     thr_args=( --threshold-by f1_plus --min-spec "${MIN_SPEC_STRESS}" ) ;;
        recall_plus) thr_args=( --threshold-by recall_plus --recall-target "${RECALL_TARGET_STRESS}" --min-spec "${MIN_SPEC_STRESS}" ) ;;
        *) echo "[ERROR] Unsupported THRESH_STRESS_MODE=${THRESH_STRESS_MODE}"; exit 2 ;;
      esac
      ;;
    *) echo "[ERROR] Unknown profile=${profile}"; exit 2 ;;
  esac

  if [[ "${RUN_BASELINE}" == "1" ]]; then
    run_block_seeds "[${human_title}] baseline" "${slug}_baseline" \
      "${COMMON_BASE[@]}" "${PREPROC_GDB[@]}" "${cv_args[@]}" "${thr_args[@]}" \
      --label-col "${label_col}" \
      "${NO_AUG[@]}"
  fi

  if [[ "${RUN_CLASSIC_MILD}" == "1" ]]; then
    run_block_seeds "[${human_title}] classic_aug (mild)" "${slug}_classic_aug" \
      "${COMMON_BASE[@]}" "${PREPROC_GDB[@]}" "${cv_args[@]}" "${thr_args[@]}" \
      --label-col "${label_col}" \
      "${CLASSIC_MILD_AUG[@]}"
  fi

  if [[ "${RUN_CLASSIC_FULL}" == "1" ]]; then
    run_block_seeds "[${human_title}] classic_full (sensitivity)" "${slug}_classic_full" \
      "${COMMON_BASE[@]}" "${PREPROC_GDB[@]}" "${cv_args[@]}" "${thr_args[@]}" \
      --label-col "${label_col}" \
      "${CLASSIC_FULL_AUG[@]}"
  fi

  if [[ "${RUN_FRDA_LITE}" == "1" ]]; then
    run_block_seeds "[${human_title}] frda_lite" "${slug}_frda_lite" \
      "${COMMON_BASE[@]}" "${PREPROC_GDB[@]}" "${cv_args[@]}" "${thr_args[@]}" \
      --label-col "${label_col}" \
      "${FRDA_LITE_AUG[@]}"
  fi

  if [[ "${RUN_META}" == "1" ]]; then
    if [[ "${RUN_BASELINE}" == "1" ]]; then
      run_block_meta "[${human_title}] baseline + meta-strat" "${slug}_baseline_meta" \
        "${COMMON_BASE[@]}" "${PREPROC_GDB[@]}" "${cv_args[@]}" "${thr_args[@]}" "${META_STRAT[@]}" \
        --label-col "${label_col}" \
        "${NO_AUG[@]}"
    fi
    if [[ "${RUN_CLASSIC_MILD}" == "1" ]]; then
      run_block_meta "[${human_title}] classic_aug + meta-strat" "${slug}_classic_aug_meta" \
        "${COMMON_BASE[@]}" "${PREPROC_GDB[@]}" "${cv_args[@]}" "${thr_args[@]}" "${META_STRAT[@]}" \
        --label-col "${label_col}" \
        "${CLASSIC_MILD_AUG[@]}"
    fi
  fi
}

# ------------------------------------------------------------
# Manifest
# ------------------------------------------------------------
cat > "${OUT_BASE}/_manifest.txt" <<EOF
run_name=${RUN_NAME}
data_gdb=${DATA_GDB}
module=${MODULE}

seeds=${GDB_SEEDS_ARR[*]}
N_JOBS=${N_JOBS}
SKIP_EXISTING=${SKIP_EXISTING}
SOFT_FAIL=${SOFT_FAIL}

spectral_min=${WN_MIN}
spectral_max=${WN_MAX}
preproc_profile=${PREPROC_PROFILE}
preproc_selected=${PREPROC_NAME}
crop_min=${CROP_MIN}
crop_max=${CROP_MAX}
drop_ranges=${DROP_RANGES}
sg_window=${SG_WINDOW}
sg_poly=${SG_POLY}
sg_deriv=${SG_DERIV}
norm=${NORM}
xscale=${XSCALE}

MODELS=${MODELS}
PROTOCOL_MAIN=${PROTOCOL_MAIN}
PROTOCOL_STRESS=${PROTOCOL_STRESS}
PLS_TUNE=${PLS_TUNE}
PLS_GRID=${PLS_GRID}

MC_ITER_MAIN=${MC_ITER_MAIN}
MC_ITER_STRESS=${MC_ITER_STRESS}
EOF

echo "=============================="
echo "GDB small-n | supervised stability experiments"
echo "Data:     ${DATA_GDB}"
echo "Output:   ${OUT_BASE}"
echo "Seeds:    ${GDB_SEEDS_ARR[*]}"
echo "Spectrum: [${WN_MIN}, ${WN_MAX}]"
echo "Preproc:  profile=${PREPROC_PROFILE} -> ${PREPROC_NAME}"
echo "          crop=${CROP_MIN}-${CROP_MAX} drop='${DROP_RANGES}' SG(${SG_WINDOW},p=${SG_POLY},d=${SG_DERIV}) norm=${NORM} xscale=${XSCALE}"
echo "Models:   ${MODELS}"
echo "PLS:      PROTOCOL_MAIN=${PROTOCOL_MAIN}, PLS_TUNE=${PLS_TUNE}, PLS_GRID=${PLS_GRID}"
echo "Main:     mc_iter=${MC_ITER_MAIN}, calib=${CALIB_MAIN}, thresh=${THRESH_MAIN_MODE}"
echo "Stress:   mc_iter=${MC_ITER_STRESS}, calib=none, thresh=${THRESH_STRESS_MODE}"
echo "Variants: baseline=${RUN_BASELINE} classic_mild=${RUN_CLASSIC_MILD} classic_full=${RUN_CLASSIC_FULL} frda_lite=${RUN_FRDA_LITE}"
echo "Tasks:    parodont=${RUN_PARODONT} anamnes=${RUN_ANAMNES} healthy_any=${RUN_HEALTHY_ANY} caries_bin=${RUN_CARIES_BIN} caries_H_path=${RUN_CARIES_H_PATH}"
echo "Meta:     RUN_META=${RUN_META} (cols=${META_STRAT_COLS}, soft=${ALLOW_META_FAIL})"
echo "Speed:    N_JOBS=${N_JOBS} SKIP_EXISTING=${SKIP_EXISTING}"
echo "=============================="

# ------------------------------------------------------------
# Run tasks
# ------------------------------------------------------------
if [[ "${RUN_PARODONT}" == "1" ]]; then
  run_task_family "T1_parodont" "y_parodont_H_vs_path" "main" "Parodont H vs pathology"
fi

if [[ "${RUN_ANAMNES}" == "1" ]]; then
  run_task_family "T2_anamnes" "y_anamnes_H_vs_path" "main" "Anamnesis H vs pathology"
fi

if [[ "${RUN_HEALTHY_ANY}" == "1" ]]; then
  run_task_family "T3_healthy_vs_any" "y_healthy_vs_any" "stress" "Healthy vs Any pathology"
fi

if [[ "${RUN_CARIES_BIN}" == "1" ]]; then
  run_task_family "T4_caries_C_vs_nonC" "y_caries_C_vs_nonC" "main" "Caries C vs non-C"
fi

if [[ "${RUN_CARIES_H_PATH}" == "1" ]]; then
  run_task_family "T5_caries_H_vs_path" "y_caries_H_vs_path" "stress" "Caries healthy vs pathology"
fi

# ------------------------------------------------------------
# Aggregation (optional)
# ------------------------------------------------------------
if [[ "${RUN_AGGREGATE}" == "1" ]]; then
  hr
  echo "[AGG] Aggregating JSON reports (if aggregator exists)..."
  if [[ -f "scripts/aggregate_gdb_smalln_reports.py" ]]; then
    "${PYTHON_BIN}" "scripts/aggregate_gdb_smalln_reports.py" \
      --reports-root "${REPORTS_ROOT}" \
      --out-dir "${REPORTS_SUMMARY_DIR}" \
      --run-name "${RUN_NAME}" || true
  else
    echo "[AGG][INFO] aggregator not found; skipped."
  fi
fi

echo
echo "ALL DONE."
echo "Reports in: ${OUT_BASE}"

if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
  echo
  echo "[WARN] Failed runs (SOFT_FAIL=1):"
  printf '  - %s\n' "${FAILED_RUNS[@]}"
fi

if [[ ${#FAILED_BLOCKS[@]} -gt 0 ]]; then
  echo
  echo "[WARN] Some optional meta-strat blocks had failures:"
  printf '  - %s\n' "${FAILED_BLOCKS[@]}"
fi