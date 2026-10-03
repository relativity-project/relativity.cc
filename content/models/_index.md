+++
title = "Models"
description = "Qwen3 and Qwen3.5 models served with SGLang-JAX and libtt on Blackhole, with measured performance, speculative decoding, and known gaps."
template = "models.html"
+++

## Supported models

These models run with SGLang-JAX's TT backend and libtt, with block-float8 (BF8) weights and BF16 activations.

<p class="perf-key"><span class="perf-key-decode">decode</span> tokens per second for a single request, higher is better<br><span class="perf-key-ttft">first token</span> milliseconds to the first token of a 215-token prompt, lower is better</p>

<table class="perf-table">
<thead><tr><th scope="col">Model</th><th scope="col">Metric</th><th scope="col">1 chip</th><th scope="col">2 chips</th><th scope="col">4 chips</th></tr></thead>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-8B">Qwen3-8B</a><span class="perf-arch">dense</span></th><td class="perf-metric">decode</td><td>38.5</td><td>60.3</td><td>85.6</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td>108</td><td>76</td><td>57</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-14B">Qwen3-14B</a><span class="perf-arch">dense</span></th><td class="perf-metric">decode</td><td>23.7</td><td>40.3</td><td>62.1</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td>163</td><td>102</td><td>75</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-32B">Qwen3-32B</a><span class="perf-arch">dense</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td>19.9</td><td>32.9</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td>203</td><td>139</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.5-9B">Qwen3.5-9B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">decode</td><td>37.2</td><td>57.1</td><td>81.6</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td>152</td><td>110</td><td>81</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.8-27B">Qwen3.8-27B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td>22.8</td><td>35.1</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td>251</td><td>186</td></tr>
</tbody>
</table>

A dash marks a configuration that does not fit in one chip's memory or that we have not run yet. Dense models use full attention in every layer; hybrid models (Qwen3.5, Qwen3.8) alternate gated DeltaNet layers with full attention.

Measured on a [QuietBox 2](../hardware/#quietbox-2) with libtt `96bed87` and SGLang-JAX `ff9b6dc`, using the [inference recipe](@/docs/inference.md) with `--tp-size` set to the chip count: greedy decoding of 128 tokens, one request at a time, median of five after two warmups.

## Batched throughput

Total decode throughput on four chips in tokens per second, with up to 16 requests running at once. Each request sends the 19-token code prompt and generates 128 tokens.

| Model | 1 request | 4 requests | 8 requests | 16 requests |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-8B | 84 | 270 | 501 | 878 |
| Qwen3-14B | 62 | 202 | 381 | 689 |
| Qwen3-32B | 32 | 110 | 207 | 376 |

At 16 requests, each request still decodes at 59 tokens/s on Qwen3-8B, 46 on Qwen3-14B and 25 on Qwen3-32B. To serve more requests at once, raise `--max-running-requests` and `--max-total-tokens` in the launch command; we used 16 and 4096.

## Speculative decoding

[DFlash](https://github.com/relativity-project/sglang-jax/blob/main/docs/features/speculative_decoding.md) speculative decoding runs on one chip with Qwen3 targets. Qwen3-8B with the [z-lab/Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) draft, in tokens per second:

| Prompt | Without draft | With draft | Speedup |
| --- | ---: | ---: | ---: |
| 19-token code request | 38.5 | 109.6 | 2.8× |
| 215-token summarization request | 37.4 | 50.1 | 1.3× |

Add these flags to the launch command:

```sh
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --grammar-backend none \
  --max-total-tokens 4096
```

## Run a model

Use the multi-chip command from the [inference recipe](@/docs/inference.md#use-several-chips) and change `--model-path` and `--tp-size`. Qwen3.5 and Qwen3.8 also need CPU PyTorch and torchvision: add `--with "torch" --with "torchvision" --index "https://download.pytorch.org/whl/cpu" --index-strategy unsafe-best-match`.

## Known gaps

- DFlash fails to compile on several chips, and does not yet support Qwen3-14B, Qwen3.5 or Qwen3.8 drafts.
- A single chip of a QuietBox needs a single-chip mesh descriptor in `TT_MESH_GRAPH_DESC_PATH`.
- Mixture-of-experts models are not covered yet.
