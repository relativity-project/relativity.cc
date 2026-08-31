+++
title = "Inference with SGLang-JAX"
description = "Serve Qwen3-8B through SGLang-JAX and libtt on a Tenstorrent Blackhole card."
weight = 30
+++

## What this path runs

SGLang-JAX provides the inference server and model runtime. JAX lowers the model through XLA, while `libtt` supplies the Tenstorrent PJRT plugin, compiler, and TT-Metal runtime as one Python wheel.

> This flow currently uses the Tenstorrent backend branch of `pcmoritz/sglang-jax`. Treat it as an experimental path and pin the revisions used for every measurement.

Start from a machine that passes the checks in [Getting started](@/docs/getting-started.md).

## Build the libtt plugin

Clone `libtt`, build the JAX plugin wheel, and keep its absolute path in the environment:

```sh
git clone https://github.com/pcmoritz/libtt.git
cd libtt

bazel build //:jax_tt_plugin_wheel
export LIBTT_WHEEL="$PWD/bazel-bin/jax_tt_plugin-0.1.0-py3-none-linux_x86_64.whl"
```

The wheel contains `libtt.so` and the initialization hook that registers the `tt` JAX platform.

## Check out SGLang-JAX

The Tenstorrent backend is currently maintained as pull request branch `#1`:

```sh
export SGLANG_JAX_DIR="$HOME/sglang-jax"
git clone https://github.com/pcmoritz/sglang-jax.git "$SGLANG_JAX_DIR"
cd "$SGLANG_JAX_DIR"

git fetch origin pull/1/head:codex/qwen3-tt-sglang
git switch codex/qwen3-tt-sglang
```

Create or activate the Python environment expected by that checkout, then install the plugin wheel:

```sh
cd "$SGLANG_JAX_DIR"
.venv/bin/python -m pip install "$LIBTT_WHEEL"
```

## Launch Qwen3-8B

This configuration serves Qwen3-8B on port `31000` with conservative token and concurrency limits:

```sh
env -u TT_METAL_RUNTIME_ROOT \
  PYTHONPATH="$SGLANG_JAX_DIR/python" \
  JAX_PLATFORMS=tt \
  JAX_USE_SHARDY_PARTITIONER=false \
  JAX_COMPILATION_CACHE_DIR=/tmp/sglang-jax-qwen3-8b-jax-cache \
  SGLANG_TT_HOST_WEIGHT_LOAD=1 \
  SGLANG_TT_OPTIMIZATION_LEVEL=1 \
  SGLANG_TT_EXPERIMENTAL_WEIGHT_DTYPE=bfp_bf8 \
  SGLANG_TT_TRACE_DECODE_ONLY=false \
  .venv/bin/python -m sgl_jax.launch_server \
    --model-path Qwen/Qwen3-8B \
    --host 127.0.0.1 \
    --port 31000 \
    --device tt \
    --dtype bfloat16 \
    --attention-backend tt \
    --max-running-requests 2 \
    --max-total-tokens 1024 \
    --max-prefill-tokens 256 \
    --chunked-prefill-size 256 \
    --page-size 32 \
    --watchdog-timeout 1200 \
    --disable-precompile \
    --skip-server-warmup \
    --disable-overlap-schedule \
    --disable-radix-cache
```

The important pieces are:

- `JAX_PLATFORMS=tt` selects the `libtt` PJRT plugin.
- `--device tt` and `--attention-backend tt` select the Tenstorrent execution paths in SGLang-JAX.
- `SGLANG_TT_TRACE_DECODE_ONLY=false` traces both fixed-shape prefill and decode instead of dispatching prefill operations individually from the host.
- `SGLANG_TT_EXPERIMENTAL_WEIGHT_DTYPE=bfp_bf8` uses the experimental lower-precision weight representation from the current backend.

## Send a request

In another terminal, ask the server for a deterministic 128-token completion:

```sh
curl -sS http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":128}}'
```

## Warm before measuring

The launch command disables precompilation and server warmup. The first requests therefore compile programs and capture traces; do not include them in steady-state measurements.

Warm every input bucket you intend to benchmark, keep prompt length and generation length fixed, and report time to first token separately from decode throughput. The current `libtt` README reports roughly 26 tokens per second on a Blackhole p150 after warmup for this configuration.

For the exact upstream command and the latest measured baseline, use the [libtt README](https://github.com/pcmoritz/libtt/blob/main/README.md).
