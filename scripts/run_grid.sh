set -euo pipefail
DATASET="covid_saliva"
N_SPLITS=5
VAL=0.2
CROP_MIN=900
CROP_MAX=1800
NORM="snv"
for NOISE in 0 0.005 0.01 0.02; do
  for SHIFT in 0 1 2 4; do
    for MIX in 0 0.2 0.4 0.6; do
      echo ">>> Run: noise=${NOISE} shift=${SHIFT} mixup=${MIX}"
      python src/train_baselines.py --dataset "${DATASET}" \
        --norm "${NORM}" --n-splits ${N_SPLITS} --val-size ${VAL} \
        --noise ${NOISE} --shift ${SHIFT} --mixup ${MIX} \
        --crop-min ${CROP_MIN} --crop-max ${CROP_MAX}
    done
  done
done
echo "Collecting summary..."
echo "Collecting summary..."
