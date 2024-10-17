#!/usr/bin/env bash
num_gpu=$3
free_gpu=$4
export CUDA_VISIBLE_DEVICES=$free_gpu
set -x

GPUS=${GPUS:-$num_gpu}
PORT=${PORT:-$5}
GPUS_PER_NODE=${GPUS_PER_NODE:-$num_gpu}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}

OUTPUT_DIR=$1
PRETRAINED_WEIGHTS=$2
PY_ARGS=${@:6}  # Any arguments from the six one are captured by this

echo "Load pretrained weights from: ${PRETRAINED_WEIGHTS}"

# train
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${PORT} --use_env \
main_actionvos.py --with_box_refine --binary --freeze_text_encoder \
--output_dir=${OUTPUT_DIR} --pretrained_weights=${PRETRAINED_WEIGHTS} ${PY_ARGS}

echo "Working path is: ${OUTPUT_DIR}"

