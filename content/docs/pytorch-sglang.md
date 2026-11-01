+++
title = "PyTorch SGLang"
description = "Serve Qwen3 with SGLang-TorchTPU, SGLang's PyTorch server for TPUs, on one Blackhole chip through torch_tpu and libtt. Experimental, with its current limitations."
weight = 35
+++

[SGLang-TorchTPU](https://github.com/sgl-project/sglang-torchtpu) is SGLang's PyTorch server for TPUs. It runs its models with PyTorch on [torch_tpu](https://github.com/google-pytorch/torch_tpu), which drives the device through the PJRT C API. libtt provides that API for Tenstorrent chips, so SGLang-TorchTPU runs on a Blackhole chip without changes to SGLang or torch_tpu.

This path is **experimental**. It serves Qwen3 dense models up to 8B on one chip, with BF16 weights. For the models and speeds on the [models page](@/models/_index.md), use [SGLang-JAX](@/docs/inference.md); see [Limitations](#limitations) for what SGLang-TorchTPU doesn't run yet.

## How it fits together

```text
SGLang-TorchTPU (PyTorch model code, torch.compile)
  → torch_tpu (PyTorch on PJRT)
  → libtt PJRT plugin, in TPU compatibility mode
  → tt-mlir compilation
  → tt-metal runtime and kernels
  → Tenstorrent device
```

`LIBTT_TPU_COMPAT=1` turns on libtt's TPU compatibility mode. libtt then reports itself as a TPU platform, so torch_tpu loads it and uses its normal compilation settings, and keeps tensors on the device in the layout programs read, as a TPU does. SGLang-TorchTPU's attention kernel is a Pallas kernel that only TPUs run; libtt replaces it with JAX and TTNN's paged attention operations when SGLang-TorchTPU loads it.

## Install

You'll need one Blackhole chip with working drivers (run the [device check](@/docs/getting-started.md#verify-jax-execution) first), [uv](https://docs.astral.sh/uv/getting-started/installation/), Git and Python 3.12.

SGLang-TorchTPU support is newer than the latest libtt release, so build the plugin wheel from libtt `main`, following the [libtt build instructions](@/docs/software-stack.md#build-and-test-libtt):

```sh
git clone https://github.com/pcmoritz/libtt.git
cd libtt
bazel build -c opt //:jax_tt_plugin_wheel
cd ..
```

Then install SGLang-TorchTPU from source, as its README describes, and the libtt wheel without its dependencies, so that SGLang-TorchTPU's pinned JAX stays:

```sh
git clone https://github.com/sgl-project/sglang-torchtpu.git
cd sglang-torchtpu
uv venv -p 3.12 .venv
cp python/pyproject_tpu.toml python/pyproject.toml
VIRTUAL_ENV=.venv uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu -e python/
VIRTUAL_ENV=.venv uv pip install --no-deps -e .
VIRTUAL_ENV=.venv uv pip install --no-deps \
  ../libtt/bazel-bin/jax_tt_plugin-0.1.5-py3-none-linux_x86_64.whl
```

## Launch Qwen3-8B

```sh
LIBTT_TPU_COMPAT=1 SGLANG_PLATFORM=tpu_oot \
TORCH_TPU_INTERNAL_XLA_OPTIONS="enable_trace=true" \
.venv/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B --host 127.0.0.1 --port 31000 \
  --device tpu --tp-size 1 --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --mem-fraction-static 0.75 --max-total-tokens 16384 \
  --enable-torch-compile --page-size 64 \
  --context-length 2048 --chunked-prefill-size 64 \
  --torch-compile-max-bs 4 --tpu-compile-bs 1 4 --tpu-compile-tokens 64 \
  --grammar-backend none
```

The first start compiles every program it needs, which takes a few minutes. The settings that matter:

- `LIBTT_TPU_COMPAT=1` turns on TPU compatibility mode (see above). `SGLANG_PLATFORM=tpu_oot` and `--device tpu` load SGLang-TorchTPU's TPU plugin and code paths.
- `TORCH_TPU_INTERNAL_XLA_OPTIONS="enable_trace=true"` records each compiled program on the device once and replays it, instead of dispatching its operations from the host every step. It doubles Qwen3-0.6B's decode speed. torch_tpu takes several options separated by spaces.
- `--kv-cache-dtype bfloat16`. SGLang-TorchTPU defaults to an FP8 KV cache, but TTNN's paged decode attention reads BF16; with FP8, every step gathers each sequence's pages and converts them on the host, about 50 times slower.
- `--mem-fraction-static 0.75`. The default of 0.4 leaves no memory for the KV cache next to Qwen3-8B's 16 GB of BF16 weights.
- `--max-total-tokens` keeps the KV pool small. Traced programs keep their buffers, and a larger pool hasn't been tested.
- `--tpu-compile-bs` and `--tpu-compile-tokens` choose the batch sizes and prefill lengths compiled at startup; requests of other shapes fail once the server is up, so cover the batch sizes you'll serve. Keep `--chunked-prefill-size` at the largest prefill length: SGLang splits longer prompts into chunks of that size, and a chunk larger than every compiled length stops the server.
- `--grammar-backend none`: structured output isn't tested.

To use one chip of a QuietBox 2, pick it by PCI address and give libtt a single-chip mesh descriptor: `TT_VISIBLE_DEVICES=0000:01:00.0 TT_MESH_GRAPH_DESC_PATH=/path/to/libtt/tests/one_chip_mesh_graph_descriptor.textproto`.

## Send a request

```sh
curl -sS http://127.0.0.1:31000/generate -H 'Content-Type: application/json' \
  -d '{"text": "The capital of the Netherlands is",
       "sampling_params": {"temperature": 0, "max_new_tokens": 64}}'
```

SGLang-TorchTPU also serves SGLang's OpenAI-compatible API on `/v1/completions` and `/v1/chat/completions`.

## Performance

Greedy decoding of 256 tokens (128 for Qwen3-8B), traced, with a BF16 KV cache, on one Blackhole chip of a [QuietBox 2](../../hardware/#quietbox-2), with libtt `main` and SGLang-TorchTPU; batch 4 is four concurrent requests:

| Model | Weights | 1 request | 4 requests |
| --- | --- | ---: | ---: |
| Qwen3-0.6B | BF16 | 89 tokens/s | 291 tokens/s (73 per request) |
| Qwen3-8B | BF16 | 19 tokens/s | 72 tokens/s (18 per request) |

Qwen3-8B reads its 16 GB of BF16 weights on every step, so it decodes at about the speed the chip's memory allows for them. SGLang-JAX stores matmul weights in BFP8 and decodes Qwen3-8B at 46 tokens/s on one chip.

## Limitations

**Models**

| Models on the [models page](@/models/_index.md) | Status with SGLang-TorchTPU | Why |
| --- | --- | --- |
| Qwen3-8B | Runs on one chip | |
| Qwen3-14B, Qwen3-32B | Doesn't run | BF16 weights don't fit on one chip, and SGLang-TorchTPU can't use several chips on Tenstorrent yet |
| Qwen3.5-9B, Qwen3.8-27B | Doesn't run | Their gated DeltaNet layers and recurrent-state copies are Pallas kernels that only TPUs run, and Qwen3.8-27B needs several chips |
| Qwen3-30B-A3B, Qwen3.5-35B-A3B, Qwen3.8-Flash-Next | Doesn't run | They need several chips |

**Features**

- **One chip only.** `--tp-size` above 1 starts one process per chip, and libtt doesn't support torch_tpu's multi-process collectives yet.
- **BF16 weights only.** SGLang-TorchTPU's FP8 weight quantization runs Pallas kernels, and libtt only converts weights to BFP8 for JAX programs that mark them, which torch_tpu's don't.
- **Speculative decoding:** EAGLE3 runs; SGLang-TorchTPU's test serves Qwen3-1.7B with the [AngelSlim/Qwen3-1.7B_eagle3](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3) draft. DFlash doesn't run yet: its draft writes the KV cache with a Pallas kernel. MTP needs Qwen3.5 or Qwen3.8 support.
- **Fixed shapes.** Only the batch sizes and prefill lengths compiled at startup run.
- **Slow starts.** Every start compiles all programs again.
- **Features built on Pallas kernels** don't run, including HiCache's KV offload to host memory, the FP8 KV cache pool, SparseCore operations and the fused MoE kernels.

If something fails, check that `LIBTT_TPU_COMPAT=1` is set in the server's environment and that the libtt wheel comes from `main`; the [software stack](@/docs/software-stack.md) page lists what to collect for a bug report.
