#!/usr/bin/env bash
set -uo pipefail

# Low-level runner used by the ep300 method switcher. CFG/MASK_TYPE/ALPHA
# determine whether PGM+D1 is enabled; AAM/D2 remains disabled.
# Protocol: d/e/f x 1..5, train on source + 8-shot target, select best.pt by
# target-domain val, then evaluate fixed target test set.

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

CONDA_SH="${CONDA_SH:-/opt/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-yolov5-7.0}"
[ -f "$CONDA_SH" ] || { echo "Conda init not found: $CONDA_SH" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV" || exit 1

TRAIN_DEV="${TRAIN_DEV:-0,1}"
TEST_DEV="${TEST_DEV:-0}"
BATCH="${BATCH:-16}"
IMG="${IMG:-640}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-10}"
SEED="${SEED:-42}"
MASK_TYPE="${MASK_TYPE:-perimeter}"
USE_MIDDLE="${USE_MIDDLE:-0}"
MIDDLE_IN_SOURCE="${MIDDLE_IN_SOURCE:-0}"
DROP_INCOMPLETE_SOURCE_BATCH="${DROP_INCOMPLETE_SOURCE_BATCH:-0}"

CFG="${CFG:-models/yolov5x_seg_SAFM_noD1.yaml}"
WEIGHTS="${WEIGHTS:-yolov5x.pt}"
HYP="${HYP:-data/hyps/rail.yaml}"
PROJ="${PROJ:-runs_pgm/a${ALPHA}/ours}"
LOGDIR="${LOGDIR:-logs/pgm/a${ALPHA}}"

mkdir -p "$PROJ" "$LOGDIR"
MASTER="$LOGDIR/master.log"
STATUS="$LOGDIR/STATUS.tsv"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER"; }

src_of() {
  case "${1:0:1}" in
    d) echo A ;; e) echo B ;; f) echo C ;; *) echo A ;;
  esac
}

middle_of() {
  case "${1:0:1}" in
    d) echo middle_domain/S_R/images ;;
    e) echo middle_domain/A_B/images ;;
    f) echo middle_domain/G_T/images ;;
    *) return 1 ;;
  esac
}

if [ "$#" -gt 0 ]; then SPLITS=("$@"); else SPLITS=(d1 d2 d3 d4 d5 e1 e2 e3 e4 e5 f1 f2 f3 f4 f5); fi

log "=================================================================="
log "Method runner starts alpha=${ALPHA} mask=${MASK_TYPE} middle_domain=${USE_MIDDLE} middle_in_source=${MIDDLE_IN_SOURCE} strict_batch=${DROP_INCOMPLETE_SOURCE_BATCH}"
log "env=${CONDA_ENV} python=$(python -c 'import sys; print(sys.executable)' 2>&1) train_dev=${TRAIN_DEV} test_dev=${TEST_DEV} batch=${BATCH} img=${IMG} workers=${WORKERS} epochs=${EPOCHS} patience=${PATIENCE} seed=${SEED}"
log "cfg=${CFG} weights=${WEIGHTS} hyp=${HYP} project=${PROJ} splits=${SPLITS[*]}"
log "=================================================================="

for sp in "${SPLITS[@]}"; do
  src="$(src_of "$sp")"
  runlog="$LOGDIR/${sp}.log"
  testlog="$PROJ/${sp}_test.log"
  best="$PROJ/${sp}/weights/best.pt"

  if [ -s "$testlog" ] && grep -qiE "^[[:space:]]*all " "$testlog"; then
    log "[$sp] already done, skip | $(grep -iE "^[[:space:]]*all " "$testlog" | tail -1 | tr -s ' ')"
    continue
  fi

  log ">>> [$sp] train start source=${src}.yaml target=${sp}.yaml alpha=${ALPHA} -> ${runlog}"
  middle_args=()
  if [ "$USE_MIDDLE" = "1" ] && [ "$MIDDLE_IN_SOURCE" = "1" ]; then
    log "!!! [$sp] USE_MIDDLE and MIDDLE_IN_SOURCE cannot both be enabled"
    printf "%s\tCONFIG_FAIL\tmiddle_modes_are_mutually_exclusive\n" "$sp" >> "$STATUS"
    continue
  elif [ "$USE_MIDDLE" = "1" ]; then
    middle_args=(--use-middle-domain --middle-data "$(middle_of "$sp")")
  elif [ "$MIDDLE_IN_SOURCE" = "1" ]; then
    middle_args=(--merge-middle-into-source --middle-data "$(middle_of "$sp")")
  fi
  strict_batch_args=()
  if [ "$DROP_INCOMPLETE_SOURCE_BATCH" = "1" ]; then
    strict_batch_args=(--drop-incomplete-source-batch)
  fi
  t0="$(date +%s)"
  python train_with_mmd_rail_seg.py \
    --data "raildatasplit/${src}.yaml" --data "raildatasplit/${sp}.yaml" \
    --cfg "$CFG" --weights "$WEIGHTS" \
    --batch-size "$BATCH" --img "$IMG" --workers "$WORKERS" --epochs "$EPOCHS" --patience "$PATIENCE" --seed "$SEED" --device "$TRAIN_DEV" \
    --hyp "$HYP" --project "$PROJ" --name "$sp" --exist-ok \
    --no-aam --mask-type "$MASK_TYPE" --alpha "$ALPHA" --beta 0 \
    "${middle_args[@]}" \
    "${strict_batch_args[@]}" \
    > "$runlog" 2>&1
  rc=$?
  dt=$(( "$(date +%s)" - t0 ))
  if [ "$rc" -ne 0 ] || [ ! -f "$best" ]; then
    log "!!! [$sp] train failed rc=${rc} best=$([ -f "$best" ] && echo yes || echo no) time=${dt}s"
    printf "%s\tTRAIN_FAIL\trc=%s_time=%ss\n" "$sp" "$rc" "$dt" >> "$STATUS"
    continue
  fi

  log "    [$sp] train done time=${dt}s, fixed-test eval -> ${testlog}"
  python val.py --task test --data "raildatasplit/${sp}.yaml" --img "$IMG" --device "$TEST_DEV" \
    --weights "$best" --project "$PROJ" --name "${sp}_test" --exist-ok \
    > "$testlog" 2>&1
  rc=$?
  row="$(grep -iE "^[[:space:]]*all " "$testlog" | tail -1 | tr -s ' ')"
  if [ "$rc" -ne 0 ]; then
    log "!!! [$sp] test failed rc=${rc}"; printf "%s\tTEST_FAIL\trc=%s\n" "$sp" "$rc" >> "$STATUS"; continue
  fi
  log "<<< [$sp] done | ${row}"
  printf "%s\tDONE\t%s\n" "$sp" "$row" >> "$STATUS"
done

log "=================================================================="
log "Method runner alpha=${ALPHA} finished. STATUS=${STATUS}"
log "=================================================================="
