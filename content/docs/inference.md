+++
title = "Inference"
description = "Serve Qwen3-8B on a Tenstorrent Blackhole card."
weight = 30
+++

In this section we describe how you can serve Qwen3-8B with upstream [SGLang-JAX](https://github.com/sgl-project/sglang-jax) as the inference server and model layer and [libtt](https://github.com/pcmoritz/libtt) as the compiler and runtime. We assume you have access to the [hardware](@/hardware/_index.md).


## Launch Qwen3-8B

This command uses `uv` to run SGLang-JAX and serve Qwen3-8B on port `31000`. Any dependencies will be set up on the fly.

```sh
JAX_PLATFORMS=tt \
JAX_USE_SHARDY_PARTITIONER=false \
uv run \
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

- `JAX_PLATFORMS=tt` selects the `libtt` PJRT plugin.
- `--device tt` and `--attention-backend tt` select the Tenstorrent execution paths in SGLang-JAX.

> The full runtime is included in the prebuilt `jax_tt_plugin` wheel. You can build your own version from [libtt](https://github.com/pcmoritz/libtt) by running `bazel build //:jax_tt_plugin_wheel`.

## Send a request

In another terminal, ask the server for a 128-token completion:

```sh
curl -sS http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":128}}'
```

Note the first two requests compile programs and capture traces so they are much slower than the steady state.

## Run the MMLU benchmark

To check the accuracy, you can run the MMLU benchmark:

```sh
git clone https://github.com/sgl-project/sglang-jax.git
cd sglang-jax
uv run \
  --with httpx --with numpy --with openai --with tqdm --with pandas \
  test/srt/run_eval.py \
  --host 127.0.0.1 \
  --port 31000 \
  --model Qwen/Qwen3-8B \
  --eval-name sglang_mmlu \
  --num-examples 10 \
  --num-threads 4 \
  --max-tokens 1024
```

This runs a small subset of 10 examples, drop the `--num-examples` parameter to run the full suite.

## Next steps

Currently we only support the Qwen3 architecture. We would also like to support the full Qwen family and Deepseek, GLM and Kimi soon (especially the flash variants). This should all happen in a way that require no or minimal changes to SGLang-JAX. It is mostly a matter of integrating the necessary kernels and compiler optimizations, a lot of the heavy lifting has already been done in [tt-metal](https://github.com/tenstorrent/tt-metal). Contributions are welcome!

We also want to support `sglang` and `vllm` going forward, this should get a lot easier with TorchTPU.
