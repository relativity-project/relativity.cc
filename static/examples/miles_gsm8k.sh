#!/bin/bash
# GRPO on GSM8K with miles on a QuietBox 2: SGLang-TorchTPU generates on chips 2 and 3, miles' torchtitan trainer
# trains on chips 0 and 1, and the trained weights reach the engine through a directory (disk-delta).
# See https://relativity.cc/docs/reinforcement-learning/.
#
# Paths (defaults under the current directory):
#   SGLANG  SGLang-TorchTPU checkout with its .venv (the PyTorch SGLang guide's install)
#   MILES   miles checkout with its .venv, next to its torchtitan checkout
#   MODEL   the model, as `hf download --local-dir` writes it
#   DATA    the GSM8K parquet files
#   RUN     logs, published weights and the engine's local checkpoint
# Recipe: ROLLOUTS, PROMPTS, SAMPLES, LR, MAX_RESPONSE, EVAL_INTERVAL (0: no evaluation).
# Further arguments go to miles' train.py.
set -u
SGLANG=${SGLANG:-$PWD/sglang-torchtpu}
MILES=${MILES:-$PWD/miles}
MODEL=${MODEL:-$PWD/models/Qwen3-0.6B}
DATA=${DATA:-$PWD/datasets/gsm8k}
RUN=${RUN:-$PWD/runs/$(date +%Y%m%d-%H%M%S)}
SGLANG_PY=${SGLANG_PY:-$SGLANG/.venv/bin/python}
MILES_PY=${MILES_PY:-$MILES/.venv/bin/python}
ROLLOUTS=${ROLLOUTS:-90} PROMPTS=${PROMPTS:-16} SAMPLES=${SAMPLES:-8} LR=${LR:-2e-6}
MAX_RESPONSE=${MAX_RESPONSE:-1024} EVAL_INTERVAL=${EVAL_INTERVAL:-30}
PORT=${PORT:-30000}
mkdir -p $RUN

stop() {
  [ -n "${ENGINE:-}" ] && kill -TERM -$ENGINE 2>/dev/null
  $(dirname $MILES_PY)/ray stop >/dev/null 2>&1
  sleep 10
  [ -n "${ENGINE:-}" ] && kill -KILL -$ENGINE 2>/dev/null
}
trap stop EXIT

# libtt gives each process its chip and joins a job's processes into one mesh.
unset TT_VISIBLE_DEVICES TT_MESH_GRAPH_DESC_PATH
export LIBTT_TPU_COMPAT=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=300

# The engine: SGLang-TorchTPU over chips 2 and 3. SGLANG_TPU_PD_COLOCATED gives tensor-parallel rank i the chip
# --base-gpu-id + i.
(
  cd $SGLANG
  export SGLANG_PLATFORM=tpu_oot JAX_PLATFORMS=cpu TORCH_TPU_INTERNAL_XLA_OPTIONS=enable_trace=true \
    SGLANG_TPU_PD_COLOCATED=1 TORCH_TPU_TOPOLOGY=2,1,1 \
    TORCH_TPU_SLICEBUILDER_ADDRESSES=localhost:41311,localhost:41312
  exec setsid $SGLANG_PY -m sglang.launch_server --model-path $MODEL --device tpu --tp-size 2 --base-gpu-id 2 \
    --host 127.0.0.1 --port $PORT --dtype bfloat16 --kv-cache-dtype bfloat16 --grammar-backend none \
    --mem-fraction-static 0.6 --enable-torch-compile --page-size 64 --context-length 2048 \
    --chunked-prefill-size 512 --tpu-compile-tokens 64 512 \
    --max-running-requests 64 --torch-compile-max-bs 64 --tpu-compile-bs 1 8 16 32 64 --watchdog-timeout 3600
) > $RUN/engine.log 2>&1 &
ENGINE=$!
until curl -sf -o /dev/null http://127.0.0.1:$PORT/health_generate; do
  kill -0 $ENGINE 2>/dev/null || { echo "The engine stopped; see $RUN/engine.log"; exit 1; }
  sleep 5
done

# The trainer: two Ray actors, one per chip. Ray's GPU IDs become torch_tpu's local ranks, so chips 0 and 1.
cd $MILES
export MILES_ACCELERATOR=tpu TORCH_TPU_TOPOLOGY=2,1,1 \
  TORCH_TPU_SLICEBUILDER_ADDRESSES=localhost:41301,localhost:41302 \
  TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0 WANDB_MODE=disabled
$(dirname $MILES_PY)/ray start --head --port 6390 --num-gpus 2 --include-dashboard=false --disable-usage-stats \
  > $RUN/ray.log 2>&1
export RAY_ADDRESS=127.0.0.1:6390
EVAL_ARGS=()
if [ "$EVAL_INTERVAL" -gt 0 ]; then
  EVAL_ARGS=(--eval-interval $EVAL_INTERVAL --eval-prompt-data gsm8k $DATA/test.parquet
    --n-samples-per-eval-prompt 1 --eval-max-response-len $MAX_RESPONSE --eval-temperature 0)
fi
$MILES_PY train.py \
  --hf-checkpoint $MODEL \
  --prompt-data $DATA/train.parquet --input-key messages --label-key label \
  --apply-chat-template --apply-chat-template-kwargs '{"enable_thinking":false}' \
  --rollout-shuffle --rm-type math \
  --num-rollout $ROLLOUTS --rollout-batch-size $PROMPTS --n-samples-per-prompt $SAMPLES \
  --rollout-max-response-len $MAX_RESPONSE --rollout-temperature 1 --global-batch-size $((PROMPTS * SAMPLES)) \
  --advantage-estimator grpo --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.28 \
  --optimizer adam --lr $LR --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
  --rollout-external-engine-addrs http://127.0.0.1:$PORT --rollout-num-gpus 2 --rollout-num-gpus-per-engine 2 \
  --update-weight-transfer-mode disk-delta --update-weight-disk-dir $RUN/weight-updates \
  --update-weight-local-checkpoint-dir $RUN/engine-checkpoint \
  --train-backend torchtitan --titan-model-name qwen3 --titan-model-flavor 0.6B \
  --distributed-backend tpu_dist --seq-length 2048 --micro-batch-size 1 --use-dynamic-batch-size \
  --max-tokens-per-gpu 2048 --gradient-checkpointing --log-probs-backend torch --titan-fsdp-gather-once \
  --actor-num-nodes 1 --actor-num-gpus-per-node 2 --num-gpus-per-node 2 \
  "${EVAL_ARGS[@]}" "$@" 2>&1 | tee $RUN/train.log
