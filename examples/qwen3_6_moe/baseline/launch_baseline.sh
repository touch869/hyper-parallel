#!/usr/bin/env bash
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Launch the reduced Qwen3.6-35B-A3B L8 baseline on one eight-die node.
#
# The recipe keeps this deployment's paths as defaults. Override them from the
# environment instead of editing the YAML:
#   MODEL_DIR    reduced L8 checkpoint directory
#   DATA_PATH    conversation parquet/JSONL with a `messages` column
#   TRAIN_ITERS  optimizer steps (default 100, the vendored recipe's value)
#   OUTPUT_DIR   checkpoint and profiler root
# Extra arguments are forwarded to scripts.train_lm as dotted overrides.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
RECIPE=${RECIPE:-${SCRIPT_DIR}/baseline_openr1_cp2.yaml}

MODEL_DIR=${MODEL_DIR:-/root/s0-l8/model-l8-real}
DATA_PATH=${DATA_PATH:-/root/s0-l8/data/openr1-math-220k-train-00000.parquet}
TRAIN_ITERS=${TRAIN_ITERS:-100}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/baseline_openr1_cp2}

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
RDZV_ID=${RDZV_ID:-qwen3_6_baseline_cp2}
PYTHON=${PYTHON:-python3}
ASCEND_ENV=${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}

# The Ascend stack must be sourced before the worker imports torch_npu.
# set_env.sh reads unset variables, so relax nounset only for the source call.
if [[ -f "${ASCEND_ENV}" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${ASCEND_ENV}"
    set -u
fi

export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}

cd "${PROJECT_ROOT}"
"${PYTHON}" -m torch.distributed.run \
    --nproc_per_node=8 \
    --rdzv_id="${RDZV_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --module scripts.train_lm \
    "${RECIPE}" \
    --model.pretrained_model_name_or_path="${MODEL_DIR}" \
    --dataset.model_assets.tokenizer.pretrained_model_name_or_path="${MODEL_DIR}" \
    --dataset.data_path="${DATA_PATH}" \
    --training.train_iters="${TRAIN_ITERS}" \
    --checkpoint.checkpoint_dir="${OUTPUT_DIR}/checkpoints" \
    --profiler.output_path="${OUTPUT_DIR}/profiling" \
    "$@"
