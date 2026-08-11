#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

usage() {
  cat <<'EOF'
Usage:
  bash run_ep300_method.sh --method METHOD [--middle-mode MODE] [options]

Methods:
  baseline       no PGM/D1, no middle domain
  pgm            PGM supervision + D1 modulation
  middle         middle domain enabled; requires --middle-mode
  pgm-middle     PGM+D1 plus middle domain; requires --middle-mode

Middle modes:
  fixed          batch12 = 10 source + 1 middle + 1 target
  pool           batch12 = 11 shuffled source/middle-pool + 1 target

Options:
  --splits CSV       default: d1,d2,d3,e1,e2,e3,f1,f2,f3
  --batch N          default: 12
  --workers N        default: 16
  --img N            default: 640
  --epochs N         default: 300
  --patience N       default: 400
  --seed N           default: 42
  --train-dev IDS    default: 0,1
  --test-dev ID      default: first id from --train-dev
  --project PATH     override the method-specific result directory
  --logdir PATH      override the method-specific log directory
  --dry-run          print the resolved configuration without training
  -h, --help         show this help

Examples:
  bash run_ep300_method.sh --method baseline --splits f2 --dry-run
  bash run_ep300_method.sh --method middle --middle-mode fixed --splits f2
  bash run_ep300_method.sh --method middle --middle-mode pool --splits d1,d2,d3
  bash run_ep300_method.sh --method pgm-middle --middle-mode pool --splits f1,f2,f3
EOF
}

METHOD=""
MIDDLE_MODE=""
SPLIT_CSV="d1,d2,d3,e1,e2,e3,f1,f2,f3"
BATCH=12
WORKERS=16
IMG=640
EPOCHS=300
PATIENCE=400
SEED=42
TRAIN_DEV="0,1"
TEST_DEV=""
PROJECT_OVERRIDE=""
LOGDIR_OVERRIDE=""
DRY_RUN=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --method) METHOD="${2:?missing value for --method}"; shift 2 ;;
    --middle-mode) MIDDLE_MODE="${2:?missing value for --middle-mode}"; shift 2 ;;
    --splits) SPLIT_CSV="${2:?missing value for --splits}"; shift 2 ;;
    --batch) BATCH="${2:?missing value for --batch}"; shift 2 ;;
    --workers) WORKERS="${2:?missing value for --workers}"; shift 2 ;;
    --img) IMG="${2:?missing value for --img}"; shift 2 ;;
    --epochs) EPOCHS="${2:?missing value for --epochs}"; shift 2 ;;
    --patience) PATIENCE="${2:?missing value for --patience}"; shift 2 ;;
    --seed) SEED="${2:?missing value for --seed}"; shift 2 ;;
    --train-dev) TRAIN_DEV="${2:?missing value for --train-dev}"; shift 2 ;;
    --test-dev) TEST_DEV="${2:?missing value for --test-dev}"; shift 2 ;;
    --project) PROJECT_OVERRIDE="${2:?missing value for --project}"; shift 2 ;;
    --logdir) LOGDIR_OVERRIDE="${2:?missing value for --logdir}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value in "$BATCH" "$WORKERS" "$IMG" "$EPOCHS" "$PATIENCE" "$SEED"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "Batch, workers, image size, epochs, patience and seed must be non-negative integers" >&2
    exit 2
  fi
done
if [ "$BATCH" -lt 3 ]; then
  echo "--batch must be at least 3" >&2
  exit 2
fi

IFS=',' read -r -a SPLITS <<< "$SPLIT_CSV"
if [ "${#SPLITS[@]}" -eq 0 ]; then
  echo "--splits must contain at least one split" >&2
  exit 2
fi
for split in "${SPLITS[@]}"; do
  if [[ ! "$split" =~ ^[def][1-5]$ ]]; then
    echo "Invalid split '$split'; expected d1-d5, e1-e5 or f1-f5" >&2
    exit 2
  fi
done

CFG="models/yolov5x_seg_SAFM_noD1.yaml"
MASK_TYPE="none"
ALPHA="0"
USE_MIDDLE=0
MIDDLE_IN_SOURCE=0
METHOD_KEY=""
DEFAULT_PROJECT=""
DEFAULT_LOGDIR=""

case "$METHOD" in
  baseline)
    [ -z "$MIDDLE_MODE" ] || { echo "baseline does not accept --middle-mode" >&2; exit 2; }
    METHOD_KEY="baseline"
    DEFAULT_PROJECT="runs_ep300/baseline/ours"
    DEFAULT_LOGDIR="logs/ep300_method/baseline"
    ;;
  pgm)
    [ -z "$MIDDLE_MODE" ] || { echo "pgm does not accept --middle-mode" >&2; exit 2; }
    METHOD_KEY="pgm_d1"
    CFG="models/yolov5x_seg_SAFM.yaml"
    MASK_TYPE="perimeter"
    ALPHA="0.05"
    DEFAULT_PROJECT="runs_ep300/pgm_d1/ours"
    DEFAULT_LOGDIR="logs/ep300_method/pgm_d1"
    ;;
  middle|pgm-middle)
    if [ "$MIDDLE_MODE" != "fixed" ] && [ "$MIDDLE_MODE" != "pool" ]; then
      echo "$METHOD requires --middle-mode fixed or --middle-mode pool" >&2
      exit 2
    fi
    if [ "$MIDDLE_MODE" = "fixed" ]; then
      USE_MIDDLE=1
    else
      MIDDLE_IN_SOURCE=1
    fi
    if [ "$METHOD" = "pgm-middle" ]; then
      CFG="models/yolov5x_seg_SAFM.yaml"
      MASK_TYPE="perimeter"
      ALPHA="0.05"
      METHOD_KEY="pgm_d1_middle_${MIDDLE_MODE}"
      if [ "$MIDDLE_MODE" = "fixed" ]; then
        DEFAULT_PROJECT="runs_ep300/middle_fixed_pgm_d1/ours"
      else
        DEFAULT_PROJECT="runs_ep300/middle_source_pool_pgm_d1/ours"
      fi
      DEFAULT_LOGDIR="logs/ep300_method/$METHOD_KEY"
    else
      METHOD_KEY="middle_${MIDDLE_MODE}"
      if [ "$MIDDLE_MODE" = "fixed" ]; then
        DEFAULT_PROJECT="runs_ep300/baseline_middle/ours"
      else
        DEFAULT_PROJECT="runs_ep300/middle_domain/ours"
      fi
      DEFAULT_LOGDIR="logs/ep300_method/$METHOD_KEY"
    fi
    ;;
  *)
    echo "--method must be baseline, pgm, middle or pgm-middle" >&2
    exit 2
    ;;
esac

PROJECT="${PROJECT_OVERRIDE:-$DEFAULT_PROJECT}"
LOGDIR="${LOGDIR_OVERRIDE:-$DEFAULT_LOGDIR}"
TEST_DEV="${TEST_DEV:-${TRAIN_DEV%%,*}}"
SOURCE_SLOTS=$((BATCH - 1 - USE_MIDDLE))

echo "method=$METHOD_KEY"
echo "domain_batch=source_or_pool:${SOURCE_SLOTS},middle:${USE_MIDDLE},target:1,total:${BATCH}"
echo "cfg=$CFG mask_type=$MASK_TYPE alpha=$ALPHA no_aam=1 beta=0"
echo "workers=$WORKERS img=$IMG epochs=$EPOCHS patience=$PATIENCE seed=$SEED"
echo "train_dev=$TRAIN_DEV test_dev=$TEST_DEV splits=${SPLITS[*]}"
echo "project=$PROJECT"
echo "logdir=$LOGDIR"

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

CONDA_ENV="${CONDA_ENV:-yolov5-7.0}" \
TRAIN_DEV="$TRAIN_DEV" TEST_DEV="$TEST_DEV" \
BATCH="$BATCH" WORKERS="$WORKERS" IMG="$IMG" \
EPOCHS="$EPOCHS" PATIENCE="$PATIENCE" SEED="$SEED" \
CFG="$CFG" MASK_TYPE="$MASK_TYPE" ALPHA="$ALPHA" \
USE_MIDDLE="$USE_MIDDLE" MIDDLE_IN_SOURCE="$MIDDLE_IN_SOURCE" \
DROP_INCOMPLETE_SOURCE_BATCH=1 \
PROJ="$PROJECT" LOGDIR="$LOGDIR" \
bash run_pgm_only.sh "${SPLITS[@]}"
