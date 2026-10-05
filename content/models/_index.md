+++
title = "Models"
description = "Qwen3, Qwen3.5 and Qwen3.8 models served with SGLang-JAX and libtt on Blackhole, with measured performance, speculative decoding, and known gaps."
template = "models.html"
+++

## Supported models

These models run on SGLang-JAX's TT backend and libtt, with weights in block-float8 (BF8) and activations in BF16.

<p class="perf-key"><span class="perf-key-decode">decode</span> tokens per second for a single request, higher is better<br><span class="perf-key-ttft">first token</span> milliseconds to the first token of a 215-token prompt, lower is better</p>

<table class="perf-table">
<thead><tr><th scope="col">Model</th><th scope="col">Metric</th><th scope="col">1 chip</th><th scope="col">2 chips</th><th scope="col">4 chips</th></tr></thead>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-8B">Qwen3-8B</a><span class="perf-arch">dense</span></th><td class="perf-metric">decode</td><td>42.9</td><td>72.4</td><td>109.6</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td>129</td><td>87</td><td>63</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-14B">Qwen3-14B</a><span class="perf-arch">dense</span></th><td class="perf-metric">decode</td><td>25.5</td><td>45.7</td><td>76.2</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td>199</td><td>122</td><td>86</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-32B">Qwen3-32B</a><span class="perf-arch">dense</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td>21.7</td><td>37.9</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td>248</td><td>163</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.5-9B">Qwen3.5-9B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">decode</td><td>39.0</td><td>62.5</td><td>93.0</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td>176</td><td>119</td><td>90</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.8-27B">Qwen3.8-27B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td>23.8</td><td>38.0</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td>292</td><td>211</td></tr>
</tbody>
</table>

A dash means the model doesn't fit on one chip or we haven't run that configuration yet. Dense models use full attention in every layer; the hybrid models (Qwen3.5, Qwen3.8) mix gated DeltaNet layers with full attention.

Measured on a [QuietBox 2](../hardware/#quietbox-2) with libtt `9dfdc65` and SGLang-JAX `a205d16`, using the [inference recipe](@/docs/inference.md) with `--tp-size` set to the chip count and overlap scheduling on: remove `--disable-overlap-schedule` from the launch command. Overlap scheduling is only faster with libtt from `4540907` on, which isn't in a release yet; with older libtt builds, keep the flag. It adds about one decode step to the time to first token. Each number is greedy decoding of 128 tokens, one request at a time, taking the median of five runs after two warmups.

## Batched throughput

Decode throughput on four chips with several requests in flight. Each request sends a 19-token code prompt and generates 128 tokens.

<p class="perf-key"><span class="perf-key-total">total</span> tokens per second across all running requests<br><span class="perf-key-decode">per user</span> tokens per second for each request</p>

<table class="perf-table">
<thead><tr><th scope="col">Model</th><th scope="col">Metric</th><th scope="col">1 request</th><th scope="col">4 requests</th><th scope="col">8 requests</th><th scope="col">16 requests</th></tr></thead>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-8B">Qwen3-8B</a><span class="perf-arch">dense</span></th><td class="perf-metric">total</td><td>106</td><td>305</td><td>551</td><td>938</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>109.5</td><td>79.1</td><td>74.3</td><td>64.8</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-14B">Qwen3-14B</a><span class="perf-arch">dense</span></th><td class="perf-metric">total</td><td>74</td><td>225</td><td>426</td><td>730</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>76.0</td><td>58.4</td><td>56.2</td><td>49.7</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-32B">Qwen3-32B</a><span class="perf-arch">dense</span></th><td class="perf-metric">total</td><td>37</td><td>119</td><td>225</td><td>399</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>37.9</td><td>31.0</td><td>29.9</td><td>27.1</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.5-9B">Qwen3.5-9B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">total</td><td>90</td><td>303</td><td>514</td><td>812</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>93.2</td><td>80.5</td><td>69.4</td><td>57.7</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.8-27B">Qwen3.8-27B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">total</td><td>37</td><td>127</td><td>226</td><td>366</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>37.9</td><td>33.5</td><td>30.6</td><td>25.2</td></tr>
</tbody>
</table>

To serve more requests at once, raise `--max-running-requests` and `--max-total-tokens` in the launch command (we used 16 and 4096). For Qwen3.5 and Qwen3.8, also set `--max-recurrent-state-size` to the same value as `--max-running-requests`; otherwise SGLang-JAX keeps a quarter of the recurrent state slots in reserve and runs at most 12 of 16 requests at once.

These batched numbers also use overlap scheduling, with libtt `9dfdc65` and SGLang-JAX `a205d16`.

## Speculative decoding

[DFlash](https://github.com/relativity-project/sglang-jax/blob/main/docs/features/speculative_decoding.md) speculative decoding works on one chip with Qwen3 models. Here is Qwen3-8B with the [z-lab/Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) draft model, in tokens per second:

| Prompt | Without draft | With draft | Speedup |
| --- | ---: | ---: | ---: |
| 19-token code request | 40.8 | 125.6 | 3.1× |
| 215-token summarization request | 40.4 | 50.2 | 1.2× |

To turn it on, add these flags to the launch command:

```sh
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --grammar-backend none \
  --max-total-tokens 4096
```

Keep `--disable-overlap-schedule` from the recipe with these flags: SGLang-JAX only runs DFlash with overlap scheduling when `--speculative-num-draft-tokens` is one more than `--speculative-num-steps`. We measured the numbers above with `--disable-overlap-schedule`, libtt `9dfdc65` and SGLang-JAX `a205d16`.

## Run a model

Start from the multi-chip command in the [inference recipe](@/docs/inference.md#use-several-chips) and change `--model-path` and `--tp-size`. Qwen3.5 and Qwen3.8 also need CPU builds of PyTorch and torchvision; add `--with "torch" --with "torchvision" --index "https://download.pytorch.org/whl/cpu" --index-strategy unsafe-best-match`.

## Known gaps

- DFlash doesn't compile on more than one chip yet, and doesn't support Qwen3-14B, Qwen3.5 or Qwen3.8 drafts.
- To use a single chip of a QuietBox, you need a single-chip mesh descriptor in `TT_MESH_GRAPH_DESC_PATH`.
- Mixture-of-experts models aren't supported yet.
