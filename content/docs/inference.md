+++
title = "Inference"
description = "Serve Qwen models with SGLang-JAX and libtt on one or more Blackhole chips."
weight = 30
+++

Serve Qwen models with [SGLang-JAX](https://github.com/relativity-project/sglang-jax) as the inference server and model implementation, and [libtt](https://github.com/pcmoritz/libtt) as the compiler and runtime. The [models page](@/models/_index.md) lists the supported models with measured performance.

You need Blackhole hardware with working system drivers, such as the [p150a dev box](../../hardware/#workstation-configuration) or a [QuietBox 2](../../hardware/#quietbox-2), plus [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git. Complete the [device check](@/docs/getting-started.md#verify-jax-execution) first.

## Launch Qwen3-8B

This command uses `uv` to install the Python dependencies and run SGLang-JAX on `127.0.0.1:31000`. Model weights download from Hugging Face on first use.

```sh
env -u TT_METAL_RUNTIME_ROOT \
JAX_PLATFORMS=tt \
JAX_USE_SHARDY_PARTITIONER=true \
uv run --no-project --python 3.12 \
  --with "sglang-jax @ git+https://github.com/relativity-project/sglang-jax.git#subdirectory=python" \
  --with "jax-tt-plugin" \
  --with "jax==0.11.1" \
  --with "jaxlib==0.11.1" \
  -m sgl_jax.launch_server \
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
  --disable-radix-cache \
  --stream-interval 1
```

The important pieces are:

- `JAX_PLATFORMS=tt` selects the libtt PJRT plugin.
- `JAX_USE_SHARDY_PARTITIONER=true` selects the partitioner the TT backend is validated with, on one chip and on several.
- `--device tt` and `--attention-backend tt` select the Tenstorrent execution paths in SGLang-JAX.
- `env -u TT_METAL_RUNTIME_ROOT` clears any override so the plugin uses its bundled runtime.

The full compiler and runtime are included in the prebuilt `jax-tt-plugin` wheel. To build your own version from libtt `main`, follow the [libtt build instructions](@/docs/software-stack.md#build-and-test-libtt), run `bazel build -c opt //:jax_tt_plugin_wheel`, and replace `--with "jax-tt-plugin"` with `--with /path/to/libtt/bazel-bin/jax_tt_plugin-0.1.0-py3-none-linux_x86_64.whl`.

## Use several chips

A QuietBox 2 has four Blackhole chips on two p300c cards. SGLang-JAX runs one process with a tensor-parallel mesh over the chips, and TT-Fabric carries the collectives. Select the chips by PCI address, remove any mesh descriptor from the environment, and pass `--tp-size`:

```sh
env -u TT_METAL_RUNTIME_ROOT -u TT_MESH_GRAPH_DESC_PATH \
TT_VISIBLE_DEVICES=0000:01:00.0,0000:02:00.0,0000:03:00.0,0000:04:00.0 \
JAX_PLATFORMS=tt \
JAX_USE_SHARDY_PARTITIONER=true \
uv run --no-project --python 3.12 \
  --with "sglang-jax @ git+https://github.com/relativity-project/sglang-jax.git#subdirectory=python" \
  --with "jax-tt-plugin" \
  --with "jax==0.11.1" \
  --with "jaxlib==0.11.1" \
  -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-14B \
  --tp-size 4 \
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
  --disable-radix-cache \
  --stream-interval 1
```

List the PCI addresses with `tt-smi -ls` and substitute your own. For two chips, list exactly two addresses and pass `--tp-size 2`: a two-chip mesh fails during fabric initialization while other chips are visible. Using one chip of a QuietBox needs a single-chip mesh descriptor; see the [known gaps](@/models/_index.md#known-gaps).

Qwen3.5 and Qwen3.8 are multimodal model classes, and SGLang-JAX loads their image and video processors even for text requests. Those processors need PyTorch and torchvision; add their CPU builds to the `uv run` command:

```sh
  --with "torch" --with "torchvision" \
  --index "https://download.pytorch.org/whl/cpu" \
  --index-strategy unsafe-best-match \
```

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
git clone https://github.com/relativity-project/sglang-jax.git
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
  --num-threads 2 \
  --max-tokens 1024
```

This runs a small subset of 10 examples to check the evaluation path. Remove `--num-examples` to run the full suite and measure accuracy. Keep `--num-threads` no higher than the server's `--max-running-requests`.

## Next steps

The [models page](@/models/_index.md) covers the Qwen3 and Qwen3.5 models we run today, speculative decoding with DFlash, and the known gaps.

We also want to support more of the Qwen family, DeepSeek, GLM, and Kimi, especially their smaller and flash variants. The goal is to reuse upstream SGLang-JAX model code with minimal changes, adding the necessary kernels and compiler support. Existing work in [tt-metal](https://github.com/tenstorrent/tt-metal) provides a starting point. Contributions are welcome through [libtt](https://github.com/pcmoritz/libtt/issues).

Support for the PyTorch-based SGLang and vLLM servers is also planned. We intend to use TorchTPU for that work once it is open-sourced; see the [framework roadmap](@/docs/software-stack.md#framework-support-and-torchtpu).
