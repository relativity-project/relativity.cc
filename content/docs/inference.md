+++
title = "Inference"
description = "Serve Qwen3-8B with SGLang-JAX and libtt on one Blackhole card."
weight = 30
+++

## Reference configuration

This recipe follows [libtt's published Qwen3-8B setup](https://github.com/pcmoritz/libtt/blob/b50ce2db8c3dbdebf1ba1818cae833dc472f34e2/README.md). It pins the libtt source and the SGLang-JAX TT integration revision, uses Python 3.12 and JAX 0.8.1, and serves on `127.0.0.1:31000`.

You need a Blackhole card with working system drivers, [uv](https://docs.astral.sh/uv/getting-started/installation/), Git, and Bazel. Complete the [device check](@/docs/getting-started.md#verify-jax-execution) first. Model weights download from Hugging Face on first use, so allow disk space and time for that download as well as the compiler build.

## Build the reference plugin

From a directory for your source checkouts:

```sh
git clone https://github.com/pcmoritz/libtt.git
cd libtt
git checkout b50ce2db8c3dbdebf1ba1818cae833dc472f34e2
bazel build //:jax_tt_plugin_wheel
export LIBTT_WHEEL="$PWD/bazel-bin/jax_tt_plugin-0.1.0-py3-none-linux_x86_64.whl"
cd ..
```

This builds the compiler and runtime into the plugin wheel. For simpler JAX experiments, [Getting started](@/docs/getting-started.md#install-the-jax-plugin) uses a prebuilt wheel; the source build here fixes the libtt revision for the serving recipe.

## Install SGLang-JAX

Continue in the same shell so `LIBTT_WHEEL` remains set:

```sh
git clone https://github.com/sgl-project/sglang-jax.git
cd sglang-jax
git fetch origin pull/1527/head
git checkout 3fc69af87cb7fb855bcdc6bee412bf8786c6a75a
uv venv --python 3.12
uv pip install --python .venv/bin/python \
  -e python \
  "jax==0.8.1" \
  "jaxlib==0.8.1" \
  "$LIBTT_WHEEL"
uv pip freeze --python .venv/bin/python > inference-requirements.txt
```

The SGLang revision comes from the [TT integration referenced by libtt](https://github.com/sgl-project/sglang-jax/pull/1527). Keep `inference-requirements.txt` with your results: the two Git revisions do not pin every transitive Python dependency.

## Launch Qwen3-8B

Run from the SGLang-JAX checkout:

```sh
env -u TT_METAL_RUNTIME_ROOT \
  JAX_PLATFORMS=tt \
  JAX_USE_SHARDY_PARTITIONER=false \
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

| Setting | Why it is here |
| --- | --- |
| `JAX_PLATFORMS=tt` | Requires the libtt PJRT backend. |
| `env -u TT_METAL_RUNTIME_ROOT` | Prevents an existing runtime-path override from selecting a different tt-metal installation. |
| `--device tt`, `--attention-backend tt` | Select the Tenstorrent execution and attention implementations. |
| `--max-running-requests 2`, `--max-total-tokens 1024` | Keep the initial concurrency and token budget at the reference recipe's settings. |
| `--disable-precompile`, `--skip-server-warmup` | Defer compilation and trace capture to incoming requests. |
| `--disable-overlap-schedule`, `--disable-radix-cache` | Preserve the reference backend configuration. Test changes to these options independently. |

## Send a request

Wait for the server to finish loading, then run this in another terminal:

```sh
curl --fail-with-body -sS http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":128}}'
```

The response contains generated text. The first requests can take substantially longer while programs compile and traces are captured. Warm each prompt-length bucket before measuring; one short prompt does not warm every input shape.

Record prompt length, generated token count, concurrency, time to first token, and decode tokens per second separately. Also record the host CPU configuration, which can affect dispatch overhead.

## Check accuracy with MMLU

Keep the server running. In another shell, enter the same SGLang-JAX checkout and run its evaluator:

```sh
.venv/bin/python test/srt/run_eval.py \
  --host 127.0.0.1 \
  --port 31000 \
  --model Qwen/Qwen3-8B \
  --eval-name mmlu \
  --num-examples 10 \
  --num-threads 2 \
  --max-tokens 1024
```

Ten examples check that the evaluation path works; they are too few for a meaningful model accuracy claim. Remove `--num-examples` for the full dataset. Keep evaluator concurrency no higher than the server's `--max-running-requests`, and save the evaluation settings and score with the package versions. Longer prompts may trigger more compilation.

## Scope and open work

This configuration covers Qwen3-8B inference on one card. It does not establish support for DeepSeek, GLM, Kimi, or the separate SGLang and vLLM PyTorch servers.

Qwen 3.5 architecture support is already in progress in [libtt #226](https://github.com/pcmoritz/libtt/pull/226). The PR adds recurrent attention kernels for SGLang-JAX and documents single-request Qwen3.5-9B serving. The recipe above remains specific to Qwen3-8B.

To bring up another architecture, isolate unsupported operations, check outputs against a reference, and measure prefill and decode separately. Include model revision, shapes, dtypes, and a small reproducer in a [libtt issue](https://github.com/pcmoritz/libtt/issues). See [training](@/docs/training.md) for the additional work needed for backward passes and optimizer updates.
