#!/usr/bin/env bash
# Leave the definitive runs going and go to sleep.
#
#   bash scripts/overnight.sh              # paper scale, the article's numbers
#   bash scripts/overnight.sh quick        # a couple of hours, to sanity-check
#
# Runs quick first when asked for paper, so that a configuration error surfaces
# in twenty minutes rather than at hour six. Set SKIP_PRECHECK=1 to go straight
# to the full run.
set -u

SCALE="${1:-paper}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"

export PYTHONUNBUFFERED=1 MPLBACKEND=Agg PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="${HERE}:${HERE}/experiments:${PYTHONPATH:-}"

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p results
MASTER="results/overnight_${SCALE}_${STAMP}.log"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${MASTER}"; }

say "==============================================================="
say " overnight run   scale=${SCALE}   host=$(hostname)"
say " python: $(python3 --version 2>&1)"
say " free disk: $(df -h . | awk 'NR==2{print $4}')"
say "==============================================================="

say "unit tests"
if ! python3 -m pytest tests -q >> "${MASTER}" 2>&1; then
  say "TESTS FAILED - stopping before wasting the night"
  exit 1
fi
say "tests passed"

if [ "${SCALE}" = "paper" ] && [ -z "${SKIP_PRECHECK:-}" ]; then
  say "pre-check at smoke scale"
  if ! bash scripts/run_all.sh smoke >> "${MASTER}" 2>&1; then
    say "SMOKE RUN FAILED - stopping"
    exit 1
  fi
  say "smoke passed, starting ${SCALE}"
fi

say "starting the ${SCALE} suite"
START=$(date +%s)
if bash scripts/run_all.sh "${SCALE}" >> "${MASTER}" 2>&1; then
  STATUS="completed"
else
  STATUS="FINISHED WITH FAILURES"
fi
END=$(date +%s)

say "${STATUS} in $(( (END-START)/3600 ))h $(( ((END-START)%3600)/60 ))m"
say "results in results/, digest in results/FINDINGS_${SCALE}.md"
du -sh results/EXP-*_"${SCALE}" 2>/dev/null | tee -a "${MASTER}"

echo ""
echo "==================== FINDINGS ===================="
cat "results/FINDINGS_${SCALE}.md" 2>/dev/null | head -80
