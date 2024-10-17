#!/usr/bin/env bash
free_gpu=$3
export CUDA_VISIBLE_DEVICES=$free_gpu
set -x

GPUS=${GPUS:-1}
PORT=${PORT:-$4}
if [ $GPUS -lt 1 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-1}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-1}

OUTPUT_DIR=$1
CHECKPOINT=$2
PY_ARGS=${@:5}  # Any arguments from the forth one are captured by this

echo "Load model weights from: ${CHECKPOINT}"

python3 inference_actionvos.py --with_box_refine --binary --freeze_text_encoder \
--output_dir=${OUTPUT_DIR} --resume=${CHECKPOINT}  ${PY_ARGS}

echo "Working path is: ${OUTPUT_DIR}"
