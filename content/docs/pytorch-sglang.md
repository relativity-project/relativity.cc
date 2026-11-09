+++
title = "PyTorch SGLang"
description = "Serve Qwen3 with SGLang-TorchTPU, SGLang's PyTorch server for TPUs, on one or several Blackhole chips through torch_tpu and libtt, with BF16 or FP8 weights. Experimental, with its current limitations and the work in progress."
weight = 35
+++

[SGLang-TorchTPU](https://github.com/sgl-project/sglang-torchtpu) is SGLang's PyTorch server for TPUs. It runs its models with PyTorch on [torch_tpu](https://github.com/google-pytorch/torch_tpu), which drives the device through the PJRT C API. libtt provides that API for Tenstorrent chips, so SGLang-TorchTPU runs on Blackhole chips without changes to SGLang or torch_tpu.

This path is **experimental**. With libtt `main` it serves the Qwen3 dense models and the Qwen3.5 and Qwen3.8 [hybrid models](#hybrid-models) with BF16 or [FP8](#fp8-weights) weights, on one chip or with tensor parallelism over several. The [models page](@/models/_index.md#sglang-torchtpu) compares its speeds with [SGLang-JAX](@/docs/inference.md)'s, which is faster and also runs the mixture-of-experts models. See [Limitations](#limitations) for what SGLang-TorchTPU doesn't run yet, and [Work in progress](#work-in-progress) for what comes next.

## How it fits together

```text
SGLang-TorchTPU (PyTorch model code, torch.compile)
  → torch_tpu (PyTorch on PJRT)
  → libtt PJRT plugin, in TPU compatibility mode
  → tt-mlir compilation
  → tt-metal runtime and kernels
  → Tenstorrent device
```

`LIBTT_TPU_COMPAT=1` turns on libtt's TPU compatibility mode:
- libtt reports itself as a TPU platform, so torch_tpu loads it and uses its normal compilation settings;
- tensors stay on the device in the layout programs read, as on a TPU;
- SGLang-TorchTPU's attention kernel, a Pallas kernel that only TPUs run, is replaced with JAX and TTNN's paged attention operations when SGLang-TorchTPU loads it.

With several chips, SGLang-TorchTPU runs one process per chip, as on a TPU host. libtt joins the processes into one mesh, and their collectives run over the chips' Ethernet links.

## Install

You'll need Blackhole chips with working drivers (run the [device check](@/docs/getting-started.md#verify-jax-execution) first), [uv](https://docs.astral.sh/uv/getting-started/installation/), Git and Python 3.12.

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

- **`LIBTT_TPU_COMPAT=1`** turns on TPU compatibility mode (see above). `SGLANG_PLATFORM=tpu_oot` and `--device tpu` load SGLang-TorchTPU's TPU plugin and code paths.
- **`TORCH_TPU_INTERNAL_XLA_OPTIONS="enable_trace=true"`** records each compiled program on the device once and replays it, instead of dispatching its operations from the host every step. It doubles Qwen3-0.6B's decode speed. torch_tpu takes several options separated by spaces.
- **`--kv-cache-dtype bfloat16`:** SGLang-TorchTPU defaults to an FP8 KV cache, but TTNN's paged decode attention reads BF16. With FP8, every step gathers each sequence's pages and converts them on the host, about 50 times slower.
- **`--mem-fraction-static 0.75`:** the default of 0.4 leaves no memory for the KV cache next to Qwen3-8B's 16 GB of BF16 weights.
- **`--max-total-tokens`** keeps the KV pool small. Traced programs keep their buffers, and a larger pool hasn't been tested.
- **`--tpu-compile-bs` and `--tpu-compile-tokens`** choose the batch sizes and prefill lengths compiled at startup. Requests of other shapes fail once the server is up, so cover the batch sizes you'll serve.
- **`--chunked-prefill-size`** must equal the largest prefill length. SGLang splits longer prompts into chunks of that size, and a chunk larger than every compiled length stops the server. To prefill long prompts in fewer chunks, also compile a longer length: with `--tpu-compile-tokens 64 256 --chunked-prefill-size 256`, Qwen3-8B on one chip returns the first token of a 219-token prompt after 0.77 s, against 1.10 s in 64-token chunks. The extra program adds about a minute to each start.
- **`--grammar-backend none`:** structured output isn't tested.

To use one chip of a QuietBox 2, pick it by PCI address and give libtt a single-chip mesh descriptor: `TT_VISIBLE_DEVICES=0000:01:00.0 TT_MESH_GRAPH_DESC_PATH=/path/to/libtt/tests/one_chip_mesh_graph_descriptor.textproto`.

## FP8 weights

SGLang-TorchTPU serves Qwen's FP8 checkpoints, such as [Qwen/Qwen3-8B-FP8](https://huggingface.co/Qwen/Qwen3-8B-FP8), and quantizes a BF16 checkpoint to FP8 when it starts with `--quantization fp8`. FP8 weights halve the bytes a decode step reads (see [Performance](#performance)). Start the server as above, with the FP8 model or `--quantization fp8`, and set `SGLANG_TPU_ENABLE_FP8_LINEAR_SMALL_M_KERNEL=1`:

```sh
LIBTT_TPU_COMPAT=1 SGLANG_PLATFORM=tpu_oot SGLANG_TPU_ENABLE_FP8_LINEAR_SMALL_M_KERNEL=1 \
TORCH_TPU_INTERNAL_XLA_OPTIONS="enable_trace=true" \
.venv/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B-FP8 --host 127.0.0.1 --port 31000 \
  --device tpu --tp-size 1 --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --mem-fraction-static 0.75 --max-total-tokens 16384 \
  --enable-torch-compile --page-size 64 \
  --context-length 2048 --chunked-prefill-size 64 \
  --torch-compile-max-bs 4 --tpu-compile-bs 1 4 --tpu-compile-tokens 64 \
  --grammar-backend none
```

With that variable, SGLang-TorchTPU sends the FP8 linear layers of decode steps to its small-batch FP8 kernel. That's a Pallas kernel, which libtt replaces when SGLang-TorchTPU loads it:
- one TT-NN operation quantizes each row of activations to FP8, as the kernel does;
- the decode matmul reads the FP8 weights and applies the activation and weight scales as it writes its result. For the MLP's gate and up projection, it also computes SiluAndMul there, for BF16 weights too.

The results are the same, bit for bit, as when the same steps run as separate TT-NN operations. Without the variable, the FP8 layers run as separate PyTorch operations, and Qwen3-8B-FP8 decodes at 37 tokens/s on four chips instead of 83.

## Several chips

Pass `--tp-size` with the number of chips, for example `--tp-size 4` on a QuietBox 2. Use `--mem-fraction-static 0.85` for Qwen3-32B. SGLang-TorchTPU then starts one process per chip, and libtt gives each process its chip and joins them into one mesh. Leave `TT_VISIBLE_DEVICES` and `TT_MESH_GRAPH_DESC_PATH` unset; libtt sets both for each process.

- **Collective buffers:** TT-NN's collectives write to their peers at the addresses of their own buffers, but separate processes don't allocate in the same order. Each process therefore reserves the same DRAM region for collective buffers when it starts: `LIBTT_SYMMETRIC_DRAM_MB` megabytes per chip, 2048 by default.
- **After a crash:** a multi-chip server that is killed can leave the chips' Ethernet links down, and the next start then hangs or fails while bringing up the fabric. Reset the chips with `tt-smi -r` before starting again.

The same support runs plain PyTorch programs with `torchrun`, one process per chip; see libtt's [torch_tpu guide](https://github.com/pcmoritz/libtt/blob/main/docs/torch-tpu.md#multi-process-tensor-parallelism).

## Hybrid models

Qwen3.5 and Qwen3.8 mix gated DeltaNet layers with full attention. SGLang-TorchTPU runs those layers with Pallas kernels, which libtt replaces with TTNN's gated delta kernels when SGLang-TorchTPU loads them; the decode kernel updates each request's recurrent state in place. Start the server as above, and add:

- `--no-enable-multimodal`, and `SGLANG_TPU_VIT_PRECOMPILE_SIZES=` (empty) in the environment, to skip the vision encoder and its warmup;
- `--max-mamba-cache-size 8`, the number of recurrent-state slots (we used 32 to run 16 requests at once);
- `--disable-radix-cache`, since prefix caching doesn't support these models yet.

For Qwen3.8-27B with FP8 weights, use [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8). With `--quantization fp8` on four chips, it stops at startup: one of its projections is 24 columns wide on each chip, and torch_tpu's FP8 matmul needs multiples of 16.

## Send a request

```sh
curl -sS http://127.0.0.1:31000/generate -H 'Content-Type: application/json' \
  -d '{"text": "The capital of the Netherlands is",
       "sampling_params": {"temperature": 0, "max_new_tokens": 64}}'
```

SGLang-TorchTPU also serves SGLang's OpenAI-compatible API on `/v1/completions` and `/v1/chat/completions`.

## Performance

Greedy decoding of 128 tokens, traced, with a BF16 KV cache, on a [QuietBox 2](../../hardware/#quietbox-2) with libtt `main` (`6b59e24`) and SGLang-TorchTPU `c1da112`:
- **1 request:** the median decode rate of three requests after a warmup.
- **4 requests:** four concurrent requests' total decode rate, the median of five batches.

**BF16 weights**

| Model | Chips | 1 request | 4 requests |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 131 tokens/s | 297 tokens/s |
| Qwen3-0.6B | 4 | 152 tokens/s | 439 tokens/s |
| Qwen3-8B | 1 | 26.5 tokens/s | 86 tokens/s |
| Qwen3-8B | 4 | 72.3 tokens/s | 223 tokens/s |
| Qwen3-14B | 4 | 45.4 tokens/s | 148 tokens/s |
| Qwen3-32B | 4 | 22.5 tokens/s | 75 tokens/s |

**FP8 weights**, with `SGLANG_TPU_ENABLE_FP8_LINEAR_SMALL_M_KERNEL=1`

| Model | Chips | 1 request | 4 requests |
| --- | ---: | ---: | ---: |
| Qwen3-8B-FP8 | 1 | 40.8 tokens/s | 120 tokens/s |
| Qwen3-8B-FP8 | 4 | 83.1 tokens/s | 245 tokens/s |
| Qwen3-8B, `--quantization fp8` | 1 | 41.6 tokens/s | 121 tokens/s |
| Qwen3-8B, `--quantization fp8` | 4 | 85.4 tokens/s | 245 tokens/s |
| Qwen3-14B, `--quantization fp8` | 4 | 59.9 tokens/s | 180 tokens/s |
| Qwen3-32B, `--quantization fp8` | 4 | 32.6 tokens/s | 99 tokens/s |

Each concurrent request returns the same text as on its own, except one of the four with Qwen3-32B and `--quantization fp8`.

Decoding reads every weight on each step, so its speed follows the bytes per weight. On one chip, Qwen3-8B reads its 16 GB of BF16 weights at about the speed the chip's memory allows, and its FP8 weights, half the bytes, decode 1.5 times as fast. SGLang-JAX stores matmul weights in BFP8, about as many bytes as FP8. On the [models page](@/models/_index.md), it decodes Qwen3-8B at 46.0 tokens/s on one chip and 122.2 on four, and Qwen3-32B at 40.7 on four. On four chips, a step's many small operations take more of the time than its weight reads, and SGLang-JAX is still faster.

## Limitations

**Models**

| Models on the [models page](@/models/_index.md) | Status with SGLang-TorchTPU on libtt `main` | Why |
| --- | --- | --- |
| Qwen3-8B, Qwen3-14B, Qwen3-32B | Run, with BF16 or FP8 weights | |
| Qwen3.5-9B, Qwen3.8-27B | Run, with BF16 or FP8 weights (see [Hybrid models](#hybrid-models)) | |
| Qwen3-30B-A3B, Qwen3.5-35B-A3B, Qwen3.8-Flash-Next | Don't run yet | SGLang-TorchTPU's fused MoE kernel is a Pallas kernel (see [Work in progress](#work-in-progress)) |

**Features**

- **No BFP8 weights.** libtt converts weights to BFP8 only for JAX programs that mark them, which torch_tpu's don't. A BF16 checkpoint therefore reads twice as many bytes per step as SGLang-JAX's BFP8 weights; use an [FP8](#fp8-weights) checkpoint or `--quantization fp8` instead.
- **Speculative decoding:** EAGLE3 runs; SGLang-TorchTPU's test serves Qwen3-1.7B with the [AngelSlim/Qwen3-1.7B_eagle3](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3) draft. MTP runs for Qwen3.5 and Qwen3.8, with the checkpoint's MTP layer as the draft (`--speculative-algorithm NEXTN`, see the [models page](@/models/_index.md#mtp)). DFlash runs on one chip, with a Qwen3 draft such as [z-lab/Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) (`--speculative-algorithm DFLASH`, see the [models page](@/models/_index.md#dflash)).
- **Fixed shapes.** Only the batch sizes and prefill lengths compiled at startup run.
- **Slow starts.** Every start compiles all programs again, since compiled programs aren't cached between runs yet.
- **Features built on Pallas kernels** don't run, including HiCache's KV offload to host memory, the FP8 KV cache pool and SparseCore operations.

If something fails, check that `LIBTT_TPU_COMPAT=1` is set in the server's environment and that the libtt wheel comes from `main`. The [software stack](@/docs/software-stack.md) page lists what to collect for a bug report.

## Work in progress

These run on development branches of libtt, not yet on `main`. Like the attention kernel, each replaces one of SGLang-TorchTPU's Pallas kernels with JAX and TTNN operations when SGLang-TorchTPU loads it, so SGLang itself stays unchanged. The numbers come from those branches on a QuietBox 2, with BF16 weights.

- **Mixture-of-experts models.** The experts run on TTNN's fused MoE kernel with BFP4 weights, as on the models page, split over the chips with `--ep-size` equal to `--tp-size`.
  - Qwen3-30B-A3B on four chips decodes 50.5 tokens/s for one request and 133 tokens/s for four.
  - It scores 0.805 on 200 MMLU questions, against about 0.80 with SGLang-JAX.
  - Qwen3.5-35B-A3B and Qwen3.8-Flash-Next are next.
- **Planned:**
  - BFP8 weights for BF16 checkpoints, as SGLang-JAX stores them;
  - caching compiled programs between starts ([libtt#463](https://github.com/pcmoritz/libtt/pull/463));
  - prefix caching for the hybrid models.
