+++
title = "Inference"
description = "Serve Qwen3-8B with SGLang-JAX and libtt on one Blackhole card."
weight = 30
+++

Serve Qwen3-8B with upstream [SGLang-JAX](https://github.com/sgl-project/sglang-jax) as the inference server and model implementation, and [libtt](https://github.com/pcmoritz/libtt) as the compiler and runtime.

You need a Blackhole card with working system drivers, such as the [p150a dev box](../../hardware/#workstation-configuration), plus [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git. Complete the [device check](@/docs/getting-started.md#verify-jax-execution) first.

## Launch Qwen3-8B

This command uses `uv` to install the Python dependencies and run SGLang-JAX on `127.0.0.1:31000`. Model weights download from Hugging Face on first use.

```sh
env -u TT_METAL_RUNTIME_ROOT \
JAX_PLATFORMS=tt \
JAX_USE_SHARDY_PARTITIONER=false \
uv run --no-project --python 3.12 \
  --with "sglang-jax @ git+https://github.com/sgl-project/sglang-jax.git#subdirectory=python" \
  --with "jax-tt-plugin" \
  --with "jax==0.8.2" \
  -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-8B \
  --host 127.0.0.1 \
  --port 31000 \
  --device tt \
  --dtype bfloat16 \
  --attention-backend tt \
  --max-running-requests 4 \
  --max-total-tokens 16384 \
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

- `JAX_PLATFORMS=tt` selects the libtt PJRT plugin.
- `--device tt` and `--attention-backend tt` select the Tenstorrent execution paths in SGLang-JAX.
- `env -u TT_METAL_RUNTIME_ROOT` clears any override so the plugin uses its bundled runtime.

The full compiler and runtime are included in the prebuilt `jax-tt-plugin` wheel. To build your own version, follow the [libtt build instructions](@/docs/software-stack.md#build-and-test-libtt) and run `bazel build //:jax_tt_plugin_wheel`.

## Send a request

Wait for the server to finish loading, then ask for a 128-token completion from another terminal:

```sh
curl --fail-with-body -sS http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":128}}'
```

The first requests compile programs and capture traces, so they are much slower than steady-state execution. Warm each prompt-length bucket before measuring performance.

## Run the MMLU benchmark

Keep the server running. In another terminal, clone SGLang-JAX and run its MMLU evaluator:

```sh
git clone https://github.com/sgl-project/sglang-jax.git
cd sglang-jax
uv run --isolated --no-project --python 3.12 \
  --with httpx --with numpy --with openai --with tqdm --with pandas \
  --with jinja2 --with requests \
  test/srt/run_eval.py \
  --host 127.0.0.1 \
  --port 31000 \
  --model Qwen/Qwen3-8B \
  --eval-name sglang_mmlu \
  --num-examples 10 \
  --num-threads 4 \
  --max-tokens 1024
```

This runs a small subset of 10 examples to check the evaluation path. Remove `--num-examples` to run the full suite and measure accuracy. Keep `--num-threads` no higher than the server's `--max-running-requests`.

## Next steps

This recipe covers Qwen3-8B. Qwen 3.5 architecture support is already in progress in [libtt #226](https://github.com/pcmoritz/libtt/pull/226), which adds recurrent attention kernels for SGLang-JAX and documents single-request Qwen3.5-9B serving.

We also want to support more of the Qwen family, DeepSeek, GLM, and Kimi, especially their smaller and flash variants. The goal is to reuse upstream SGLang-JAX model code with minimal changes, adding the necessary kernels and compiler support. Existing work in [tt-metal](https://github.com/tenstorrent/tt-metal) provides a starting point. Contributions are welcome through [libtt](https://github.com/pcmoritz/libtt/issues).

Support for the PyTorch-based SGLang and vLLM servers is also planned. We intend to use TorchTPU for that work once it is open-sourced; see the [framework roadmap](@/docs/software-stack.md#framework-support-and-torchtpu).
