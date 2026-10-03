+++
title = "Inference"
description = "Serve Qwen models with SGLang-JAX and libtt on one or more Blackhole chips."
weight = 30
+++

This guide serves Qwen models with [SGLang-JAX](https://github.com/relativity-project/sglang-jax) as the server and model code, and [libtt](https://github.com/pcmoritz/libtt) as the compiler and runtime. The [models page](@/models/_index.md) shows which models work and how fast they run.

You'll need Blackhole hardware with working drivers, such as the [p150a dev box](../../hardware/#workstation-configuration) or a [QuietBox 2](../../hardware/#quietbox-2), plus [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git. Run the [device check](@/docs/getting-started.md#verify-jax-execution) first.

## Launch Qwen3-8B

This command uses `uv` to install the dependencies and starts SGLang-JAX on `127.0.0.1:31000`. The weights download from Hugging Face the first time you run it.

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

The Tenstorrent-specific settings are:

- `JAX_PLATFORMS=tt` loads the libtt PJRT plugin.
- `JAX_USE_SHARDY_PARTITIONER=true` uses the partitioner the TT backend is tested with, on one chip or several.
- `--device tt` and `--attention-backend tt` turn on SGLang-JAX's Tenstorrent code paths.
- `env -u TT_METAL_RUNTIME_ROOT` clears any override, so the plugin uses its bundled runtime.

The prebuilt `jax-tt-plugin` wheel includes the full compiler and runtime. To use your own build of libtt `main` instead, follow the [libtt build instructions](@/docs/software-stack.md#build-and-test-libtt), run `bazel build -c opt //:jax_tt_plugin_wheel`, and replace `--with "jax-tt-plugin"` with `--with /path/to/libtt/bazel-bin/jax_tt_plugin-0.1.0-py3-none-linux_x86_64.whl`.

## Use several chips

A QuietBox 2 has four Blackhole chips on two p300c cards. SGLang-JAX runs a single process that shards the model across the chips with tensor parallelism, and TT-Fabric handles the collectives. Pick the chips by PCI address, clear any mesh descriptor from the environment, and set `--tp-size`:

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

Find your chips' PCI addresses with `tt-smi -ls` and use those. For two chips, list exactly two addresses and pass `--tp-size 2`; if other chips are visible, fabric initialization fails. Running on a single chip of a QuietBox needs a single-chip mesh descriptor; see [known gaps](@/models/_index.md#known-gaps).

Qwen3.5 and Qwen3.8 are multimodal, and SGLang-JAX loads their image and video processors even for text-only requests. Those processors need PyTorch and torchvision, so add the CPU builds to the `uv run` command:

```sh
  --with "torch" --with "torchvision" \
  --index "https://download.pytorch.org/whl/cpu" \
  --index-strategy unsafe-best-match \
```

## Send a request

Once the server has finished loading, request a 128-token completion from another terminal:

```sh
curl --fail-with-body -sS http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":128}}'
```

The first few requests compile programs and capture traces, so they're much slower than later ones. Before you measure performance, send a warmup request for each prompt length you plan to test.

## Run the MMLU benchmark

Leave the server running. In another terminal, clone SGLang-JAX and run its MMLU evaluator:

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

This runs 10 examples, which is enough to check that evaluation works end to end. Drop `--num-examples` to run the full suite and get a real accuracy number. Keep `--num-threads` at or below the server's `--max-running-requests`.

## Next steps

The [models page](@/models/_index.md) covers the Qwen3, Qwen3.5 and Qwen3.8 models we run today, speculative decoding with DFlash, and known gaps.

Next, we want to support more of the Qwen family, DeepSeek, GLM and Kimi, especially their smaller and flash variants. The plan is to reuse upstream SGLang-JAX model code with as few changes as possible, adding kernels and compiler support where needed. Existing work in [tt-metal](https://github.com/tenstorrent/tt-metal) is a good starting point. Contributions are welcome; start with an issue on [libtt](https://github.com/pcmoritz/libtt/issues).

We also plan to support the PyTorch-based SGLang and vLLM servers, using TorchTPU once it's open-sourced; see the [framework roadmap](@/docs/software-stack.md#framework-support-and-torchtpu).
