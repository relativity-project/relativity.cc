+++
title = "Models"
description = "Qwen3 and Qwen3.5 models served with SGLang-JAX and libtt on Blackhole, with measured performance, speculative decoding, and known gaps."
template = "models.html"
+++

## Supported models

These models run with SGLang-JAX's TT backend and libtt, using the [inference recipe](@/docs/inference.md). Weights are stored as block-float8 (BF8) and activations as BF16. Decode and prefill are traced, so steady-state requests replay recorded device programs.

Decode rate for a single request, in tokens per second:

| Model | Architecture | 1 chip | 2 chips | 4 chips |
| --- | --- | ---: | ---: | ---: |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | dense | 38.5 | 60.3 | 85.6 |
| [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | dense | 23.7 | — | 62.1 |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | dense | — | — | 32.9 |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | hybrid | 37.2 | — | 81.6 |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | hybrid | — | — | 35.1 |

Time to first token for a 215-token prompt, in milliseconds:

| Model | 1 chip | 2 chips | 4 chips |
| --- | ---: | ---: | ---: |
| Qwen3-8B | 108 | 76 | 57 |
| Qwen3-14B | 163 | — | 75 |
| Qwen3-32B | — | — | 139 |
| Qwen3.5-9B | 152 | — | 81 |
| Qwen3.8-27B | — | — | 186 |

A dash marks a configuration we did not run. Decode rates are for a single request, which is the latency-bound case; the [recipe](@/docs/inference.md) allows two concurrent requests. Rates vary by a few percent with the prompt (for example, 79.9 tokens/s for Qwen3-8B on four chips after the 215-token prompt), and with the host CPU's power profile.

Dense models are transformers with full attention in every layer. Qwen3.5 and Qwen3.8 are hybrid: they alternate gated DeltaNet layers, a recurrent linear attention, with full attention layers. libtt runs their decode step with dedicated recurrent-attention kernels; their prefill uses chunked recurrent attention.

## How we measured

- **Hardware:** a [QuietBox 2](../hardware/#quietbox-2), four Blackhole chips on two p300c cards.
- **Software:** libtt [`main`](https://github.com/pcmoritz/libtt) at `96bed87`, built from source with `bazel build -c opt //:jax_tt_plugin_wheel`; SGLang-JAX [`main`](https://github.com/relativity-project/sglang-jax) at `ff9b6dc`; JAX and jaxlib 0.11.1; Python 3.12.
- **Server:** the [inference recipe](@/docs/inference.md), with `--tp-size` set to the chip count. One-chip runs on the QuietBox add a single-chip mesh descriptor (see [known gaps](#known-gaps)).
- **Requests:** greedy decoding of 128 tokens with `ignore_eos`, one request at a time, streamed. For each prompt, two warmup requests, then the median of five. The decode rate counts the tokens after the first one over the time after the first one; time to first token is measured at the client.
- **Prompts:** a 5-token completion prompt, a 19-token code request, and a 215-token summarization request. The decode table shows the 5-token prompt's rate; the 19-token prompt is within 1% of it in every configuration.

## Speculative decoding

SGLang-JAX's TT backend supports [DFlash](https://github.com/relativity-project/sglang-jax/blob/main/docs/features/speculative_decoding.md) speculative decoding: a small diffusion draft model proposes a block of tokens, and the target model verifies the block in one step. Today it runs on one chip only, with Qwen3 targets.

Qwen3-8B on one chip with the [z-lab/Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) draft, decode rate in tokens per second:

| Prompt | Without draft | With draft | Speedup |
| --- | ---: | ---: | ---: |
| 19-token code request | 38.5 | 109.6 | 2.8× |
| 215-token summarization request | 37.4 | 50.1 | 1.3× |

The speedup depends on how predictable the continuation is. Across both prompts, the target accepted 3.72 draft tokens per verification step on average. Add these flags to the launch command:

```sh
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --grammar-backend none \
  --max-total-tokens 4096
```

We ran DFlash with `--max-total-tokens 4096`. Drafting and verification are greedy.

Other pairings do not work yet:

- **Qwen3-14B:** the published 14B draft, [deepseek-ai/dflash_qwen3_14b_block7](https://huggingface.co/deepseek-ai/dflash_qwen3_14b_block7), predicts from the anchor position. That layout needs `--speculative-sample-from-anchor`, which the TT backend rejects. Run as plain DFlash, it accepts 1.02 tokens per step and decodes at 18.2 tokens/s, slower than without a draft.
- **Qwen3.5 and Qwen3.8:** SGLang-JAX's Qwen3.5 model does not yet expose the hidden states DFlash drafts read, so the server fails at startup with [z-lab/Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash). The Qwen3.8-27B drafts, such as [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), use the DFlash2 architecture, which SGLang-JAX does not implement.
- **Several chips:** on four chips, DFlash fails to compile; see the first known gap.

## Run a model

Use the multi-chip command from the [inference recipe](@/docs/inference.md#use-several-chips) and change `--model-path` and `--tp-size`:

| Model | `--model-path` | `--tp-size` | Extra packages |
| --- | --- | ---: | --- |
| Qwen3-8B | `Qwen/Qwen3-8B` | 1, 2 or 4 | |
| Qwen3-14B | `Qwen/Qwen3-14B` | 1 or 4 | |
| Qwen3-32B | `Qwen/Qwen3-32B` | 4 | |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | 1 or 4 | PyTorch and torchvision (CPU) |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B` | 4 | PyTorch and torchvision (CPU) |

For example, Qwen3.8-27B on a QuietBox 2:

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
  --with "torch" --with "torchvision" \
  --index "https://download.pytorch.org/whl/cpu" \
  --index-strategy unsafe-best-match \
  -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3.8-27B \
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

The first requests compile the model and capture traces, so they are much slower than steady state. Warm each prompt-length bucket before measuring.

## Known gaps

These are open issues in SGLang-JAX, libtt or the recipe, as of the revisions above.

- **DFlash on several chips fails.** On four chips, the draft's KV-cache write is not partitioned for tensor parallelism: the `paged_update_cache` call in the draft-extend program receives each chip's slice (2 KV heads for Qwen3-8B on four chips) but declares the full 8-head cache as its output, and compilation fails.
- **DFlash for Qwen3-14B, Qwen3.5 and Qwen3.8.** As above: anchor-layout drafts are rejected on TT, the Qwen3.5 model class lacks the hidden-state capture DFlash needs, and DFlash2 drafts are not implemented.
- **One chip of a QuietBox needs a mesh descriptor.** The two chips on a p300c card are fabric-linked, so tt-metal requires a single-chip mesh graph descriptor in `TT_MESH_GRAPH_DESC_PATH`; without it, startup fails with `Custom fabric mesh graph descriptor path must be specified for CUSTOM cluster type`. The recipe covers one p150a and the multi-chip meshes, not this case.
- **Qwen3.5 and Qwen3.8 need PyTorch.** Their Hugging Face processors require PyTorch and torchvision even for text-only serving, and SGLang-JAX does not install them.
- **Repeated requests are not always identical on four chips.** For Qwen3-8B on four chips, two of the three prompts gave two different continuations across five identical greedy requests; Qwen3.5-9B on four chips did so for one prompt. One and two chips were repeatable. Outputs also differ between chip counts, and between plain and DFlash decoding, where tokens are nearly tied: BF8 weights and a different reduction order change the rounding.
- **Batched throughput is not measured here.** The recipe limits the server to two concurrent requests and 1,024 cached tokens; larger batches and longer contexts are untested in these numbers.
- **Mixture-of-experts models are not covered**, for example Qwen3-30B-A3B and Qwen3.5-35B-A3B.
- **The published plugin predates these results.** `jax-tt-plugin` 0.1.0 on PyPI is older than libtt `96bed87`; until 0.1.1 is published, build the wheel from libtt `main` as described in the [recipe](@/docs/inference.md).
