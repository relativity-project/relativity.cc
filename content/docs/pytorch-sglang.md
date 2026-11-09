+++
title = "PyTorch SGLang"
description = "Serve Qwen3 with SGLang-TorchTPU, SGLang's PyTorch server for TPUs, on one or several Blackhole chips through torch_tpu and libtt, with BF16 or FP8 weights. Experimental, with its current limitations and the work in progress."
weight = 35
+++

[SGLang-TorchTPU](https://github.com/sgl-project/sglang-torchtpu) is SGLang's PyTorch server for TPUs. It runs its models with PyTorch on [torch_tpu](https://github.com/google-pytorch/torch_tpu), which drives the device through the PJRT C API. libtt provides that API for Tenstorrent chips, so SGLang-TorchTPU runs on Blackhole chips without changes to SGLang or torch_tpu.

This path is **experimental**. With libtt 0.2.0 it serves the Qwen3 dense models and the Qwen3.5 and Qwen3.8 [hybrid models](#hybrid-models) with BF16 or [FP8](#fp8-weights) weights, on one chip or with tensor parallelism over several. The [models page](@/models/_index.md#sglang-torchtpu) has its speeds next to those of [SGLang-JAX](@/docs/inference.md), which also runs the mixture-of-experts models. See [Limitations](#limitations) for what SGLang-TorchTPU doesn't run yet, and [Work in progress](#work-in-progress) for what comes next.

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

Install SGLang-TorchTPU from source, as its README describes, at the commit we tested with libtt 0.2.0. Then install the libtt plugin from PyPI, without its dependencies so that SGLang-TorchTPU's pinned JAX stays:

```sh
git clone https://github.com/sgl-project/sglang-torchtpu.git
cd sglang-torchtpu
git checkout 5191118
uv venv -p 3.12 .venv
cp python/pyproject_tpu.toml python/pyproject.toml
echo "torch_tpu==0.2.0" > overrides.txt
VIRTUAL_ENV=.venv uv pip install --no-sources --override overrides.txt \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu -e python/
VIRTUAL_ENV=.venv uv pip install --no-deps -e .
VIRTUAL_ENV=.venv uv pip install --no-deps "jax-tt-plugin==0.2.0"
```

SGLang-TorchTPU pins a torch_tpu nightly from a package registry that asks for credentials. The override installs [torch_tpu 0.2.0](https://pypi.org/project/torch-tpu/0.2.0/) from PyPI instead, the version libtt 0.2.0 is tested with, and `--no-sources` makes uv look for it there.

To try a newer libtt, build the plugin wheel from `main` with `bazel build -c opt //:jax_tt_plugin_wheel`, following the [libtt build instructions](@/docs/software-stack.md#build-and-test-libtt), and install `bazel-bin/jax_tt_plugin-*.whl` in place of `jax-tt-plugin==0.2.0`.

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

The first start compiles every program it needs, which takes several minutes: about eight for Qwen3-8B-FP8 on one chip. SGLang-TorchTPU keeps the compiled programs in torch.compile's cache, in `~/.cache/sglang` (set `SGLANG_CACHE_DIR` to move it), and the next start with the same model and settings loads them, in about a minute and a half. Programs cached by another libtt build aren't used; torch.compile logs `fx graph cache unable to load compiled graph` when it skips them.

The settings that matter:

- **`LIBTT_TPU_COMPAT=1`** turns on TPU compatibility mode (see above). `SGLANG_PLATFORM=tpu_oot` and `--device tpu` load SGLang-TorchTPU's TPU plugin and code paths.
- **`TORCH_TPU_INTERNAL_XLA_OPTIONS="enable_trace=true"`** records each compiled program on the device once and replays it, instead of dispatching its operations from the host every step. It doubles Qwen3-0.6B's decode speed. torch_tpu takes several options separated by spaces.
- **`--kv-cache-dtype bfloat16`:** SGLang-TorchTPU defaults to an FP8 KV cache, but TTNN's paged decode attention reads BF16. With FP8, every step gathers each sequence's pages and converts them on the host, about 50 times slower.
- **`--mem-fraction-static 0.75`:** the default of 0.4 leaves no memory for the KV cache next to Qwen3-8B's 16 GB of BF16 weights.
- **`--max-total-tokens`** keeps the KV pool small. Traced programs keep their buffers, and a larger pool hasn't been tested.
- **`--tpu-compile-bs` and `--tpu-compile-tokens`** choose the batch sizes and prefill lengths compiled at startup. Requests of other shapes fail once the server is up, so cover the batch sizes you'll serve.
- **`--chunked-prefill-size`** must equal the largest prefill length. SGLang splits longer prompts into chunks of that size, and a chunk larger than every compiled length stops the server. To prefill long prompts in fewer chunks, also compile a longer length: with `--tpu-compile-tokens 64 256 --chunked-prefill-size 256`, Qwen3-8B on one chip returns the first token of a 219-token prompt after 0.77 s, against 1.10 s in 64-token chunks. The extra program adds about a minute to each start.
- **`--grammar-backend none`:** structured output isn't tested.

To use one chip of a QuietBox 2, pick it by PCI address and give libtt a one-chip mesh descriptor. Write the descriptor once:

```sh
cat > one_chip.textproto <<'EOF'
mesh_descriptors {
  name: "M0"
  arch: BLACKHOLE
  device_topology { dims: [1, 1] dim_types: [LINE, LINE] }
  host_topology { dims: [1, 1] }
  channels { count: 1 policy: RELAXED }
}
top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
EOF
```

Then add `TT_VISIBLE_DEVICES=0000:01:00.0 TT_MESH_GRAPH_DESC_PATH=$PWD/one_chip.textproto` to the launch command's environment.

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

The results are the same, bit for bit, as when the same steps run as separate TT-NN operations. Without the variable, the FP8 layers run as separate PyTorch operations: with libtt `main` at `6b59e24`, Qwen3-8B-FP8 then decoded 37 tokens/s on four chips instead of 83.

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

For Qwen3.8-27B with FP8 weights, use [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8). With `--quantization fp8` on four chips, it stops at startup: one of its projections is 24 columns wide on each chip, and torch_tpu's FP8 matmul needs multiples of 16. This command serves Qwen3.8-27B-FP8 on the four chips of a QuietBox 2:

```sh
LIBTT_TPU_COMPAT=1 SGLANG_PLATFORM=tpu_oot SGLANG_TPU_ENABLE_FP8_LINEAR_SMALL_M_KERNEL=1 \
SGLANG_TPU_VIT_PRECOMPILE_SIZES= \
TORCH_TPU_INTERNAL_XLA_OPTIONS="enable_trace=true" \
.venv/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen3.8-27B-FP8 --host 127.0.0.1 --port 31000 \
  --device tpu --tp-size 4 --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --mem-fraction-static 0.85 --max-total-tokens 16384 \
  --enable-torch-compile --page-size 64 \
  --context-length 2048 --chunked-prefill-size 64 \
  --torch-compile-max-bs 4 --tpu-compile-bs 1 4 --tpu-compile-tokens 64 \
  --no-enable-multimodal --max-mamba-cache-size 8 --disable-radix-cache \
  --grammar-backend none
```

To decode with the checkpoint's MTP layer as the draft, add `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`. A single request then decodes about twice as fast (see the [models page](@/models/_index.md#mtp)).

## Send a request

```sh
curl -sS http://127.0.0.1:31000/generate -H 'Content-Type: application/json' \
  -d '{"text": "The capital of the Netherlands is",
       "sampling_params": {"temperature": 0, "max_new_tokens": 64}}'
```

SGLang-TorchTPU also serves SGLang's OpenAI-compatible API on `/v1/completions` and `/v1/chat/completions`.

## Performance

The [models page](@/models/_index.md#sglang-torchtpu) has SGLang-TorchTPU's speeds with FP8 weights for the Qwen3 dense models and the hybrid models: single requests on one, two and four chips, sampling, batched requests, MTP and DFlash.

With BF16 weights and the [launch command above](#launch-qwen3-8b), on a [QuietBox 2](../../hardware/#quietbox-2) with libtt 0.2.0 and the same torch_tpu and SGLang-TorchTPU as the models page, measured with the models page's [benchmark script](@/models/_index.md#how-we-measure):
- **1 request:** a single greedy request's decode rate after its first token.
- **4 requests:** four requests at once, their tokens per second from sending them until the last one finishes, prefills included.

| Model | Chips | 1 request | 4 requests |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 155 tokens/s | 508 tokens/s |
| Qwen3-0.6B | 4 | 200 tokens/s | 655 tokens/s |
| Qwen3-8B | 1 | 27.3 tokens/s | 99.6 tokens/s |
| Qwen3-8B | 4 | 88.0 tokens/s | 296 tokens/s |

Decoding reads every weight on each step, so its speed follows the bytes per weight. On one chip, Qwen3-8B reads its 16 GB of BF16 weights at about the speed the chip's memory allows, and its FP8 weights, half the bytes, decode 1.6 times as fast (44.0 tokens/s on the models page). On four chips, each chip reads a quarter of the weights, and a step's many small operations take more of its time.

## Limitations

**Models**

| Models on the [models page](@/models/_index.md) | Status with SGLang-TorchTPU on libtt 0.2.0 | Why |
| --- | --- | --- |
| Qwen3-8B, Qwen3-14B, Qwen3-32B | Run, with BF16 or FP8 weights | |
| Qwen3.5-9B, Qwen3.8-27B | Run, with BF16 or FP8 weights (see [Hybrid models](#hybrid-models)) | |
| Qwen3-30B-A3B, Qwen3.5-35B-A3B, Qwen3.8-Flash-Next | Don't run yet | SGLang-TorchTPU's fused MoE kernel is a Pallas kernel (see [Work in progress](#work-in-progress)) |

**Features**

- **No BFP8 weights.** libtt converts weights to BFP8 only for JAX programs that mark them, which torch_tpu's don't. A BF16 checkpoint therefore reads twice as many bytes per step as SGLang-JAX's BFP8 weights; use an [FP8](#fp8-weights) checkpoint or `--quantization fp8` instead.
- **Speculative decoding:** EAGLE3 runs; SGLang-TorchTPU's test serves Qwen3-1.7B with the [AngelSlim/Qwen3-1.7B_eagle3](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3) draft. MTP runs for Qwen3.5 and Qwen3.8, with the checkpoint's MTP layer as the draft (`--speculative-algorithm NEXTN`, see the [models page](@/models/_index.md#mtp)). DFlash runs on one chip, with a Qwen3 draft such as [z-lab/Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) (`--speculative-algorithm DFLASH`, see the [models page](@/models/_index.md#dflash)).
- **Fixed shapes.** Only the batch sizes and prefill lengths compiled at startup run.
- **Slow first starts.** The first start with a model and its settings compiles every program, which takes several minutes; later starts load them from the cache.
- **Features built on Pallas kernels** don't run, including HiCache's KV offload to host memory, the FP8 KV cache pool and SparseCore operations.

If something fails, check that `LIBTT_TPU_COMPAT=1` is set in the server's environment and that `VIRTUAL_ENV=.venv uv pip show jax-tt-plugin` reports version 0.2.0 or newer. The [software stack](@/docs/software-stack.md) page lists what to collect for a bug report.

## Work in progress

These run on development branches of libtt, not yet on `main`. Like the attention kernel, each replaces one of SGLang-TorchTPU's Pallas kernels with JAX and TTNN operations when SGLang-TorchTPU loads it, so SGLang itself stays unchanged. The numbers come from those branches on a QuietBox 2, with BF16 weights.

- **Mixture-of-experts models.** The experts run on TTNN's fused MoE kernel with BFP4 weights, as on the models page, split over the chips with `--ep-size` equal to `--tp-size`.
  - Qwen3-30B-A3B on four chips decodes 50.5 tokens/s for one request and 133 tokens/s for four.
  - It scores 0.805 on 200 MMLU questions, against about 0.80 with SGLang-JAX.
  - Qwen3.5-35B-A3B and Qwen3.8-Flash-Next are next.
- **Planned:**
  - BFP8 weights for BF16 checkpoints, as SGLang-JAX stores them;
  - prefix caching for the hybrid models.
