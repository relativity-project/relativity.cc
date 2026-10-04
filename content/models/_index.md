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

A dash means the model doesn't fit on one chip or we haven't run that configuration yet. Dense models use full attention in every layer; the hybrid models (Qwen3.5, Qwen3.8) mix gated DeltaNet layers with full attention.

Measured on a [QuietBox 2](../hardware/#quietbox-2) with libtt `96bed87` and SGLang-JAX `ff9b6dc`, using the [inference recipe](@/docs/inference.md) with `--tp-size` set to the chip count. Each number is greedy decoding of 128 tokens, one request at a time, taking the median of five runs after two warmups.

## Batched throughput

Decode throughput on four chips with several requests in flight. Each request sends a 19-token code prompt and generates 128 tokens.

<p class="perf-key"><span class="perf-key-total">total</span> tokens per second across all running requests<br><span class="perf-key-decode">per user</span> tokens per second for each request</p>

<table class="perf-table">
<thead><tr><th scope="col">Model</th><th scope="col">Metric</th><th scope="col">1 request</th><th scope="col">4 requests</th><th scope="col">8 requests</th><th scope="col">16 requests</th></tr></thead>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-8B">Qwen3-8B</a><span class="perf-arch">dense</span></th><td class="perf-metric">total</td><td>93</td><td>298</td><td>541</td><td>922</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>95.7</td><td>77.2</td><td>71.9</td><td>63.8</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-14B">Qwen3-14B</a><span class="perf-arch">dense</span></th><td class="perf-metric">total</td><td>67</td><td>219</td><td>413</td><td>730</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>68.4</td><td>56.7</td><td>54.7</td><td>50.1</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-32B">Qwen3-32B</a><span class="perf-arch">dense</span></th><td class="perf-metric">total</td><td>34</td><td>117</td><td>221</td><td>400</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>34.8</td><td>30.5</td><td>29.3</td><td>27.1</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.5-9B">Qwen3.5-9B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">total</td><td>90</td><td>308</td><td>529</td><td>830</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>93.5</td><td>81.8</td><td>71.6</td><td>59.2</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.8-27B">Qwen3.8-27B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">total</td><td>37</td><td>128</td><td>230</td><td>373</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>37.9</td><td>33.7</td><td>31.1</td><td>25.7</td></tr>
</tbody>
</table>

To serve more requests at once, raise `--max-running-requests` and `--max-total-tokens` in the launch command (we used 16 and 4096). For Qwen3.5 and Qwen3.8, also set `--max-recurrent-state-size` to the same value as `--max-running-requests`; otherwise SGLang-JAX keeps a quarter of the recurrent state slots in reserve and runs at most 12 of 16 requests at once.

These batched numbers use overlap scheduling: remove `--disable-overlap-schedule` from the launch command. Overlap scheduling is only faster with libtt `cba0e0d`, which isn't in a release yet; with older libtt builds, keep the flag. It adds about one decode step to the time to first token. The numbers use libtt `cba0e0d` and SGLang-JAX `a205d16`.

## Speculative decoding

[DFlash](https://github.com/relativity-project/sglang-jax/blob/main/docs/features/speculative_decoding.md) speculative decoding works on one chip with Qwen3 models. Here is Qwen3-8B with the [z-lab/Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) draft model, in tokens per second:

| Prompt | Without draft | With draft | Speedup |
| --- | ---: | ---: | ---: |
| 19-token code request | 38.5 | 109.6 | 2.8× |
| 215-token summarization request | 37.4 | 50.1 | 1.3× |

To turn it on, add these flags to the launch command:

```sh
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --grammar-backend none \
  --max-total-tokens 4096
```

## Run a model

Start from the multi-chip command in the [inference recipe](@/docs/inference.md#use-several-chips) and change `--model-path` and `--tp-size`. Qwen3.5 and Qwen3.8 also need CPU builds of PyTorch and torchvision; add `--with "torch" --with "torchvision" --index "https://download.pytorch.org/whl/cpu" --index-strategy unsafe-best-match`.

## Known gaps

- DFlash doesn't compile on more than one chip yet, and doesn't support Qwen3-14B, Qwen3.5 or Qwen3.8 drafts.
- To use a single chip of a QuietBox, you need a single-chip mesh descriptor in `TT_MESH_GRAPH_DESC_PATH`.
- Mixture-of-experts models aren't supported yet.
