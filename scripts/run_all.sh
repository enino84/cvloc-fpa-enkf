#!/usr/bin/env bash
# Run the whole suite, or a selection of it.
#
#   bash scripts/run_all.sh              # scale from $SCALE, default smoke
#   bash scripts/run_all.sh paper
#   EXPERIMENTS="exp07_kradii exp10_cholesky" bash scripts/run_all.sh quick
#   SHARD_INDEX=1 SHARD_COUNT=4 bash scripts/run_all.sh paper
#
# Experiments are independent: each writes to its own directory and a failure
# in one does not stop the others. The exit code is non-zero if any failed, so
# a CI job or a cluster wrapper still notices.
set -u

SCALE="${1:-${SCALE:-smoke}}"
export SCALE
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${HERE}:${HERE}/experiments:${PYTHONPATH:-}"
RESULTS="${RESULTS_DIR:-${HERE}/results}"
mkdir -p "${RESULTS}"

DEFAULT_EXPERIMENTS="exp00_tuning \
exp01_landscape \
exp02_baselines \
exp03_cvproxy \
exp04_meta \
exp05_budget \
exp06_ensemble \
exp07_kradii \
exp08_hetero \
exp09_cycles \
exp10_cholesky"

SELECTED="${EXPERIMENTS:-$DEFAULT_EXPERIMENTS}"

SUFFIX=""
if [ "${SHARD_COUNT:-1}" != "1" ]; then
  SUFFIX="_shard${SHARD_INDEX:-0}"
fi
LOG="${RESULTS}/run_${SCALE}${SUFFIX}.log"

echo "===============================================================" | tee -a "${LOG}"
echo " CV-localization suite   scale=${SCALE}   $(date -Is)"           | tee -a "${LOG}"
echo " shard ${SHARD_INDEX:-0}/${SHARD_COUNT:-1}   methods='${METHODS:-all}'" | tee -a "${LOG}"
echo " experiments: ${SELECTED}"                                       | tee -a "${LOG}"
echo "===============================================================" | tee -a "${LOG}"

# EXP-00 must run first when it is in the selection: it writes
# results/frozen_params.json, which every other experiment reads.
FAILED=""
for exp in ${SELECTED}; do
  echo ""                                                    | tee -a "${LOG}"
  echo ">>> ${exp}  ($(date +%H:%M:%S))"                     | tee -a "${LOG}"
  if python3 "${HERE}/experiments/${exp}.py" "${SCALE}" 2>&1 | tee -a "${LOG}"; then
    echo "<<< ${exp} ok"                                     | tee -a "${LOG}"
  else
    echo "<<< ${exp} FAILED"                                 | tee -a "${LOG}"
    FAILED="${FAILED} ${exp}"
  fi
done

if [ "${SHARD_COUNT:-1}" = "1" ]; then
  echo ""                                                    | tee -a "${LOG}"
  echo ">>> assembling tables"                               | tee -a "${LOG}"
  python3 "${HERE}/scripts/make_tables.py" "${SCALE}" 2>&1   | tee -a "${LOG}"
fi

echo ""                                                      | tee -a "${LOG}"
if [ -n "${FAILED}" ]; then
  echo "FAILED:${FAILED}"                                    | tee -a "${LOG}"
  exit 1
fi
echo "all experiments completed   $(date -Is)"               | tee -a "${LOG}"
