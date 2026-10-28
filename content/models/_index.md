+++
title = "Models"
description = "Qwen3, Qwen3.5 and Qwen3.8 models, dense, hybrid and mixture-of-experts, served with SGLang-JAX and libtt on Blackhole, with performance numbers, speculative decoding, and known gaps."
template = "models.html"
+++

## Supported models

These models run on SGLang-JAX's TT backend and libtt. Matmul weights, including the LM head, are stored in BFP8, Tenstorrent's 8-bit block floating point format, in which each group of 16 values shares an exponent. The mixture-of-experts models' experts are the exception: TTNN's fused MoE kernel stores them in BFP4, with 4 bits per value. Everything else (embeddings, norm weights, activations and the KV cache) is BF16.

<p class="perf-key"><span class="perf-key-decode">decode</span> tokens per second for a single greedy request without speculative decoding, higher is better<br><span class="perf-key-sampled">sampled</span> the same request sampled at temperature 0.7, without top-p or top-k (see <a href="#sampling">Sampling</a>)<br><span class="perf-key-ttft">first token</span> milliseconds to the first token of a 215-token prompt, lower is better</p>

<table class="perf-table">
<thead><tr><th scope="col">Model</th><th scope="col">Metric</th><th scope="col">1 chip</th><th scope="col">2 chips</th><th scope="col">4 chips</th></tr></thead>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3-8B">Qwen3-8B</a><span class="perf-arch">dense</span></th><td class="perf-metric">decode</td><td>44.5</td><td>77.7</td><td>119.7</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td>41.7</td><td>68.8</td><td>98.3</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td>116</td><td>78</td><td>56</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3-14B">Qwen3-14B</a><span class="perf-arch">dense</span></th><td class="perf-metric">decode</td><td>26.1</td><td>48.0</td><td>82.3</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td>25.1</td><td>44.3</td><td>71.6</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td>184</td><td>113</td><td>77</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3-32B">Qwen3-32B</a><span class="perf-arch">dense</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td>22.3</td><td>40.4</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td class="perf-none">—</td><td>21.5</td><td>37.6</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td>233</td><td>149</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3.5-9B">Qwen3.5-9B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">decode</td><td>39.8</td><td>65.8</td><td>101.2</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td>36.3</td><td>55.5</td><td>77.3</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td>168</td><td>115</td><td>83</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3.8-27B">Qwen3.8-27B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td>24.5</td><td>41.2</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td class="perf-none">—</td><td>22.9</td><td>36.6</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td>287</td><td>196</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3-30B-A3B">Qwen3-30B-A3B</a><span class="perf-arch">MoE</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td>101.1</td><td>124.7</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td class="perf-none">—</td><td>86.3</td><td>101.7</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td>98</td><td>72</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3.5-35B-A3B">Qwen3.5-35B-A3B</a><span class="perf-arch">MoE</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td>74.8</td><td>92.4</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td class="perf-none">—</td><td>61.9</td><td>72.1</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td>150</td><td>111</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Qwen3.8-Flash-Next</a><span class="perf-arch">MoE</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td class="perf-none">—</td><td>39.3</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td class="perf-none">—</td><td class="perf-none">—</td><td>34.5</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td class="perf-none">—</td><td>815</td></tr>
</tbody>
</table>

A dash means the model doesn't fit on one chip or we haven't run that configuration yet. Dense models use full attention in every layer; the hybrid models (Qwen3.5, Qwen3.8) mix gated DeltaNet layers with full attention. The mixture-of-experts (MoE) models run each token through a few of their experts, with the experts split over the chips. Qwen3-30B-A3B and Qwen3.5-35B-A3B use 8 of 128 and 256 experts, about 3B active parameters, and need at least two chips; Qwen3.8-Flash-Next uses 10 of 512 and needs four. Qwen3.5-35B-A3B and Qwen3.8-Flash-Next are also hybrid.

Measured on a [QuietBox 2](../hardware/#quietbox-2) with libtt 0.1.4 and SGLang-JAX `fc0a170`, using the [inference recipe](@/docs/inference.md) with `--tp-size` set to the chip count, overlap scheduling on (remove `--disable-overlap-schedule` from the launch command) and `JAX_DEFAULT_PRNG_IMPL=unsafe_rbg` (see [Sampling](#sampling)). Each number decodes 128 tokens, one request at a time and without [speculative decoding](#speculative-decoding), taking the median of five runs after two warmups.

## Sampling

Set `JAX_DEFAULT_PRNG_IMPL=unsafe_rbg` in the launch command's environment when requests sample:

```sh
export JAX_DEFAULT_PRNG_IMPL=unsafe_rbg
```

It makes JAX draw its random bits with libtt's native generator. JAX's default Threefry generator adds about 25 ms to every sampled token on Blackhole for the Qwen3 models' 151,936-token vocabulary, and 40 ms for Qwen3.8-Flash-Next's 248,320. That halves sampled decode on one chip: 20.4 instead of 41.7 tokens per second for Qwen3-8B.

With it, sampling at temperature 0.7 adds 1.5 to 1.8 ms per token for the Qwen3 models and 2.4 to 3.5 ms for Qwen3.5 and Qwen3.8, whose vocabulary is larger. Its share grows as decode gets faster: from 4% for Qwen3-32B on two chips to 24% for Qwen3.5-9B on four. `top_p` and `top_k` each add a search over the vocabulary to every step, and a request that sets only `temperature` runs neither. Qwen3-8B, one request, tokens per second:

| Sampling parameters | 1 chip | 4 chips |
| --- | ---: | ---: |
| greedy (`temperature: 0`) | 44.5 | 119.7 |
| `temperature: 0.7` | 41.7 | 98.3 |
| `temperature: 0.7, top_p: 0.95` | 36.4 | 73.3 |
| `temperature: 0.7, top_k: 20` | 36.6 | 74.3 |
| `temperature: 0.7, top_p: 0.95, top_k: 20` | 32.5 | 58.9 |

In a batch, a search runs for every request as long as any request in the batch needs it. `min_p` filters without a search.

## Batched throughput

Decode throughput on four chips with several requests in flight. Each request sends a 19-token code prompt and generates 128 tokens.

<p class="perf-key"><span class="perf-key-total">total</span> tokens per second across all running requests<br><span class="perf-key-decode">per user</span> tokens per second for each request</p>

<table class="perf-table">
<thead><tr><th scope="col">Model</th><th scope="col">Metric</th><th scope="col">1 request</th><th scope="col">4 requests</th><th scope="col">8 requests</th><th scope="col">16 requests</th></tr></thead>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-8B">Qwen3-8B</a><span class="perf-arch">dense</span></th><td class="perf-metric">total</td><td>116</td><td>400</td><td>716</td><td>1106</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>119.3</td><td>105.3</td><td>96.8</td><td>78.8</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-14B">Qwen3-14B</a><span class="perf-arch">dense</span></th><td class="perf-metric">total</td><td>80</td><td>282</td><td>515</td><td>884</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>82.2</td><td>73.9</td><td>69.3</td><td>62.5</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-32B">Qwen3-32B</a><span class="perf-arch">dense</span></th><td class="perf-metric">total</td><td>39</td><td>140</td><td>264</td><td>468</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>40.3</td><td>36.9</td><td>35.8</td><td>32.6</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.5-9B">Qwen3.5-9B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">total</td><td>98</td><td>337</td><td>567</td><td>885</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>101.2</td><td>88.9</td><td>76.1</td><td>62.3</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.8-27B">Qwen3.8-27B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">total</td><td>40</td><td>140</td><td>248</td><td>394</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>41.3</td><td>36.8</td><td>33.3</td><td>26.9</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-30B-A3B">Qwen3-30B-A3B</a><span class="perf-arch">MoE</span></th><td class="perf-metric">total</td><td>120</td><td>384</td><td>668</td><td>989</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>124.5</td><td>103.2</td><td>91.5</td><td>74.6</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.5-35B-A3B">Qwen3.5-35B-A3B</a><span class="perf-arch">MoE</span></th><td class="perf-metric">total</td><td>89</td><td>296</td><td>473</td><td>720</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>92.2</td><td>78.6</td><td>63.8</td><td>50.7</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Qwen3.8-Flash-Next</a><span class="perf-arch">MoE</span></th><td class="perf-metric">total</td><td>36</td><td>93</td><td>134</td><td>160</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>38.6</td><td>25.9</td><td>19.6</td><td>11.0</td></tr>
</tbody>
</table>

To serve more requests at once, raise `--max-running-requests` and `--max-total-tokens` in the launch command (we used 16 and 4096). For Qwen3.5 and Qwen3.8, also set `--max-recurrent-state-size` to the same value as `--max-running-requests`; otherwise SGLang-JAX keeps a quarter of the recurrent state slots in reserve and runs at most 12 of 16 requests at once.

These batched numbers are greedy and also use overlap scheduling, with the same libtt and SGLang-JAX as above.

## Speculative decoding

[DFlash](https://github.com/relativity-project/sglang-jax/blob/main/docs/features/speculative_decoding.md) speculative decoding works on one chip with Qwen3 models. Here is Qwen3-8B with the [z-lab/Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) draft model, in tokens per second:

| Prompt | Without draft | With draft | Speedup |
| --- | ---: | ---: | ---: |
| 19-token code request | 42.4 | 122.4 | 2.9× |
| 215-token summarization request | 41.6 | 73.5 | 1.8× |

To turn it on, add these flags to the launch command:

```sh
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --grammar-backend none \
  --max-total-tokens 4096
```

Keep `--disable-overlap-schedule` from the recipe with these flags: SGLang-JAX only runs DFlash with overlap scheduling when `--speculative-num-draft-tokens` is one more than `--speculative-num-steps`. We measured the numbers above with `--disable-overlap-schedule`, libtt 0.1.4 and the same SGLang-JAX as above.

## Run a model

Start from the multi-chip command in the [inference recipe](@/docs/inference.md#use-several-chips) and change `--model-path` and `--tp-size`. If your requests sample, also set `JAX_DEFAULT_PRNG_IMPL=unsafe_rbg` (see [Sampling](#sampling)). The mixture-of-experts models also need `--moe-backend fused`. Qwen3.5 and Qwen3.8 also need CPU builds of PyTorch and torchvision; add `--with "torch" --with "torchvision" --index "https://download.pytorch.org/whl/cpu" --index-strategy unsafe-best-match`.

Qwen3.8-Flash-Next also needs safetensors with [pull request #862](https://github.com/safetensors/safetensors/pull/862) until it is merged, so that its weights stay in the memory-mapped checkpoint instead of being copied into RAM. SGLang-JAX pins an older safetensors, which `uv run` can't override, so install everything into an environment with the override instead (building safetensors from source needs a Rust toolchain):

```sh
echo 'safetensors @ git+https://github.com/pcmoritz/safetensors.git@numpy-zero-copy#subdirectory=bindings/python' > overrides.txt
uv venv --python 3.12 flash-next
uv pip install --python flash-next/bin/python \
  "sglang-jax @ git+https://github.com/relativity-project/sglang-jax.git#subdirectory=python" \
  "jax-tt-plugin" "jax==0.11.1" "jaxlib==0.11.1" "torch" "torchvision" \
  --index "https://download.pytorch.org/whl/cpu" \
  --index-strategy unsafe-best-match \
  --override overrides.txt
```

Then run the multi-chip command with `flash-next/bin/python -m sgl_jax.launch_server` in place of `uv run ... -m sgl_jax.launch_server`, with `--model-path Qwen/Qwen3.8-Flash-Next` and `--tp-size 4`.

## Known gaps

- DFlash doesn't compile on more than one chip yet, and doesn't support Qwen3-14B, Qwen3.5 or Qwen3.8 drafts.
- To use a single chip of a QuietBox, you need a single-chip mesh descriptor in `TT_MESH_GRAPH_DESC_PATH`.
- Qwen3.8-Flash-Next's first request after a start takes about two minutes, while its weights are read from the checkpoint and prepared on the chips.
- Qwen3.8-Flash-Next's QSA layers attend densely, which matches the model only while a request's context is at most 2,051 tokens.
