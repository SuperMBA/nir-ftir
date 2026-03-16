#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

DATA="data/processed/gdb_smalln.parquet"
FACTORS="Gender,Age_factor,caries_factor,Parodont,Anamnes_factor"

for PRE in paper_full paper_low amide3; do
  for SEED in 0 1 2 3 4; do
    OUT="reports/pca_r2/gdb_${PRE}_nomix_seed${SEED}"
    python scripts/pca_dimdesc_r2.py \
      --data-path "${DATA}" \
      --outdir "${OUT}" \
      --factors "${FACTORS}" \
      --preproc-profile "${PRE}" \
      --classic-profile mild \
      --use-mixup 0 \
      --seed "${SEED}"
  done
done

python scripts/aggregate_dimdesc_r2.py
echo "[DONE] See reports/pca_r2/dimdesc_r2_best_pc_per_factor.csv"