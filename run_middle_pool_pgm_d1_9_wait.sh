#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

CONDA_SH="${CONDA_SH:-/opt/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-yolov5-7.0}"
PROJECT="${PROJECT:-runs_ep300/middle_source_pool_pgm_d1/ours}"
LOGROOT="${LOGROOT:-logs/ep300_middle_source_pool_pgm_d1}"
WAIT_PIDS=("$@")

mkdir -p "$PROJECT" "$LOGROOT/lane01" "$LOGROOT/lane23"
MASTER="$LOGROOT/wait_and_launch.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER"; }

log "Queue created; waiting for current jobs: ${WAIT_PIDS[*]:-none}"
for pid in "${WAIT_PIDS[@]}"; do
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done
  log "Waited process $pid has exited"
done

gpu_or_training_busy() {
  local gpu_pids
  if gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
    [ -n "${gpu_pids//[[:space:]]/}" ]
  else
    pgrep -f '[p]ython .*train_with_mmd_rail_seg.py|[p]ython .*val.py' >/dev/null
  fi
}

# Avoid racing any GPU job that started while this queue was waiting.
while gpu_or_training_busy; do
  log "GPU compute or train/val processes are still active; checking again in 60 seconds"
  sleep 60
done

log "GPUs released by current experiments; launching 9 middle-pool + PGM+D1 runs"
COMMON=(
  MIDDLE_IN_SOURCE=1 USE_MIDDLE=0 BATCH=12 IMG=640
  EPOCHS=300 PATIENCE=400 SEED=42 ALPHA=0.05 MASK_TYPE=perimeter
  CFG=models/yolov5x_seg_SAFM.yaml
  PROJ="$PROJECT"
)

env "${COMMON[@]}" TRAIN_DEV=0,1 TEST_DEV=0 LOGDIR="$LOGROOT/lane01" \
  CONDA_ENV="$CONDA_ENV" CONDA_SH="$CONDA_SH" \
  bash run_pgm_only.sh d1 d2 d3 e1 e2 > "$LOGROOT/lane01.nohup.log" 2>&1 &
lane01_pid=$!

env "${COMMON[@]}" TRAIN_DEV=2,3 TEST_DEV=2 LOGDIR="$LOGROOT/lane23" \
  CONDA_ENV="$CONDA_ENV" CONDA_SH="$CONDA_SH" \
  bash run_pgm_only.sh e3 f1 f2 f3 > "$LOGROOT/lane23.nohup.log" 2>&1 &
lane23_pid=$!

printf '%s\n' "$lane01_pid" > "$LOGROOT/lane01.pid"
printf '%s\n' "$lane23_pid" > "$LOGROOT/lane23.pid"
log "Started lane01 pid=$lane01_pid and lane23 pid=$lane23_pid"

rc=0
wait "$lane01_pid" || rc=1
log "lane01 completed"
wait "$lane23_pid" || rc=1
log "lane23 completed"
log "All 9 requested runs finished; rc=$rc"
exit "$rc"
