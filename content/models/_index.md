+++
title = "Models"
description = "Qwen3, Qwen3.5 and Qwen3.8 models, dense, hybrid and mixture-of-experts, served with SGLang-JAX or SGLang-TorchTPU and libtt on Blackhole, with performance numbers, speculative decoding, and known gaps."
template = "models.html"
+++

## Supported models

These models run on SGLang-JAX's TT backend and libtt. Matmul weights, including the LM head, are stored in BFP8, Tenstorrent's 8-bit block floating point format, in which each group of 16 values shares an exponent. The mixture-of-experts models' experts are the exception: TTNN's fused MoE kernel stores them in BFP4, with 4 bits per value. Everything else (embeddings, norm weights, activations and the KV cache) is BF16. The dense and hybrid models also run on [SGLang-TorchTPU](#sglang-torchtpu), SGLang's PyTorch server, with FP8 weights.

<p class="perf-key"><span class="perf-key-decode">decode</span> tokens per second after the first token, for a single greedy request without speculative decoding, higher is better<br><span class="perf-key-sampled">sampled</span> the same request sampled at temperature 0.7, without top-p or top-k (see <a href="#sampling">Sampling</a>)<br><span class="perf-key-ttft">first token</span> milliseconds to the first token of a 215-token prompt, lower is better</p>

<table class="perf-table">
<thead><tr><th scope="col">Model</th><th scope="col">Metric</th><th scope="col">1 chip</th><th scope="col">2 chips</th><th scope="col">4 chips</th></tr></thead>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3-8B">Qwen3-8B</a><span class="perf-arch">dense</span></th><td class="perf-metric">decode</td><td>46.2</td><td>80.4</td><td>124.7</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td>44.4</td><td>74.7</td><td>111.2</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td>115</td><td>76</td><td>54</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3-14B">Qwen3-14B</a><span class="perf-arch">dense</span></th><td class="perf-metric">decode</td><td>26.7</td><td>49.0</td><td>83.3</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td>26.1</td><td>46.8</td><td>77.1</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td>182</td><td>112</td><td>77</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3-32B">Qwen3-32B</a><span class="perf-arch">dense</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td>22.4</td><td>40.4</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td class="perf-none">—</td><td>22.0</td><td>38.9</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td>231</td><td>148</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3.5-9B">Qwen3.5-9B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">decode</td><td>43.4</td><td>74.6</td><td>117.1</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td>41.0</td><td>66.6</td><td>98.3</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td>163</td><td>111</td><td>80</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3.8-27B">Qwen3.8-27B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td>26.7</td><td>46.0</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td class="perf-none">—</td><td>25.6</td><td>42.8</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td>279</td><td>185</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3-30B-A3B">Qwen3-30B-A3B</a><span class="perf-arch">MoE</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td>103.6</td><td>125.9</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td class="perf-none">—</td><td>94.2</td><td>112.1</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td>97</td><td>71</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3.5-35B-A3B">Qwen3.5-35B-A3B</a><span class="perf-arch">MoE</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td>88.5</td><td>108.6</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td class="perf-none">—</td><td>77.8</td><td>92.0</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td>145</td><td>106</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Qwen3.8-Flash-Next</a><span class="perf-arch">MoE</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td class="perf-none">—</td><td>42.5</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td class="perf-none">—</td><td class="perf-none">—</td><td>39.3</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td class="perf-none">—</td><td>812</td></tr>
</tbody>
</table>

A dash means the model doesn't fit on one chip or we haven't run that configuration yet. Dense models use full attention in every layer; the hybrid models (Qwen3.5, Qwen3.8) mix gated DeltaNet layers with full attention. The mixture-of-experts (MoE) models run each token through a few of their experts, with the experts split over the chips. Qwen3-30B-A3B and Qwen3.5-35B-A3B use 8 of 128 and 256 experts, about 3B active parameters, and need at least two chips; Qwen3.8-Flash-Next uses 10 of 512 and needs four. Qwen3.5-35B-A3B and Qwen3.8-Flash-Next are also hybrid.

Measured on a [QuietBox 2](../hardware/#quietbox-2) with libtt 0.2.0 and SGLang-JAX `6282ec6`, using the [inference recipe](@/docs/inference.md) with `--tp-size` set to the chip count, overlap scheduling on (remove `--disable-overlap-schedule` from the launch command) and `JAX_DEFAULT_PRNG_IMPL=unsafe_rbg` (see [Sampling](#sampling)). Each number decodes 128 tokens, one request at a time and without [speculative decoding](#speculative-decoding), taking the median of five runs after two warmups. [How we measure](#how-we-measure) has the prompts, the timing and a script that takes the same measurements.

## Sampling

Set `JAX_DEFAULT_PRNG_IMPL=unsafe_rbg` in the launch command's environment when requests sample:

```sh
export JAX_DEFAULT_PRNG_IMPL=unsafe_rbg
```

It makes JAX draw its random bits with libtt's native generator. JAX's default Threefry generator adds about 18 ms to every sampled token on Blackhole for the Qwen3 models' 151,936-token vocabulary, and 29 ms for Qwen3.8-Flash-Next's 248,320. That slows sampled decode on one chip from 44.4 to about 24 tokens per second for Qwen3-8B.

With it, sampling at temperature 0.7 adds 0.8 to 1.0 ms per token for the Qwen3 models and 1.3 to 2.0 ms for Qwen3.5 and Qwen3.8, whose vocabulary is larger. Its share grows as decode gets faster: from 2% for Qwen3-32B on two chips to 16% for Qwen3.5-9B on four. `top_p` and `top_k` each add a search over the vocabulary to every step, and a request that sets only `temperature` runs neither. Qwen3-8B, one request, tokens per second:

| Sampling parameters | 1 chip | 4 chips |
| --- | ---: | ---: |
| greedy (`temperature: 0`) | 46.2 | 124.7 |
| `temperature: 0.7` | 44.4 | 111.2 |
| `temperature: 0.7, top_p: 0.95` | 39.3 | 84.0 |
| `temperature: 0.7, top_k: 20` | 39.3 | 84.0 |
| `temperature: 0.7, top_p: 0.95, top_k: 20` | 35.1 | 67.1 |

In a batch, a search runs for every request as long as any request in the batch needs it. `min_p` filters without a search.

## Batched throughput

Decode throughput on four chips with several requests in flight. Each request sends a 19-token code prompt and generates 128 tokens. Total throughput includes the time the requests' prefills take, so it's lower than the number of requests times the per-user rate (see [How we measure](#how-we-measure)).

<p class="perf-key"><span class="perf-key-total">total</span> tokens per second across all requests, from sending them until the last one finishes, prefills included<br><span class="perf-key-decode">per user</span> each request's decode rate after its first token, the median over requests</p>

<table class="perf-table">
<thead><tr><th scope="col">Model</th><th scope="col">Metric</th><th scope="col">1 request</th><th scope="col">4 requests</th><th scope="col">8 requests</th><th scope="col">16 requests</th></tr></thead>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-8B">Qwen3-8B</a><span class="perf-arch">dense</span></th><td class="perf-metric">total</td><td>120</td><td>406</td><td>727</td><td>1084</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>123.2</td><td>108.0</td><td>98.5</td><td>78.3</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-14B">Qwen3-14B</a><span class="perf-arch">dense</span></th><td class="perf-metric">total</td><td>81</td><td>283</td><td>517</td><td>865</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>82.6</td><td>74.3</td><td>69.5</td><td>61.7</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-32B">Qwen3-32B</a><span class="perf-arch">dense</span></th><td class="perf-metric">total</td><td>39</td><td>140</td><td>262</td><td>460</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>40.2</td><td>36.8</td><td>35.6</td><td>31.7</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.5-9B">Qwen3.5-9B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">total</td><td>113</td><td>389</td><td>654</td><td>967</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>116.8</td><td>104.7</td><td>88.2</td><td>68.8</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.8-27B">Qwen3.8-27B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">total</td><td>45</td><td>160</td><td>280</td><td>438</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>45.9</td><td>42.1</td><td>37.9</td><td>30.9</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-30B-A3B">Qwen3-30B-A3B</a><span class="perf-arch">MoE</span></th><td class="perf-metric">total</td><td>118</td><td>383</td><td>648</td><td>946</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>122.6</td><td>103.6</td><td>90.2</td><td>70.3</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.5-35B-A3B">Qwen3.5-35B-A3B</a><span class="perf-arch">MoE</span></th><td class="perf-metric">total</td><td>104</td><td>339</td><td>559</td><td>797</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>108.1</td><td>91.8</td><td>76.1</td><td>56.6</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Qwen3.8-Flash-Next</a><span class="perf-arch">MoE</span></th><td class="perf-metric">total</td><td>39</td><td>99</td><td>144</td><td>169</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>41.9</td><td>27.7</td><td>21.3</td><td>11.6</td></tr>
</tbody>
</table>

To serve more requests at once, raise `--max-running-requests` and `--max-total-tokens` in the launch command (we used 16 and 4096). For Qwen3.5 and Qwen3.8, also set `--max-recurrent-state-size` to the same value as `--max-running-requests`; otherwise SGLang-JAX keeps a quarter of the recurrent state slots in reserve and runs at most 12 of 16 requests at once.

These batched numbers are greedy and also use overlap scheduling, with the same libtt and SGLang-JAX as above.

## SGLang-TorchTPU

[SGLang-TorchTPU](@/docs/pytorch-sglang.md), SGLang's PyTorch server for TPUs, also runs the dense and hybrid models, through torch_tpu and libtt. It's experimental, and it doesn't run the mixture-of-experts models yet. Here are the same measurements with FP8 weights: Qwen's FP8 checkpoints for Qwen3-8B, Qwen3-14B, Qwen3-32B and Qwen3.8-27B, and `--quantization fp8` for Qwen3.5-9B, which has no FP8 checkpoint. FP8 weights take about as many bytes as SGLang-JAX's BFP8 weights above.

<p class="perf-key"><span class="perf-key-decode">decode</span> tokens per second for a single greedy request, higher is better<br><span class="perf-key-sampled">sampled</span> the same request sampled at temperature 0.7, without top-p or top-k<br><span class="perf-key-ttft">first token</span> milliseconds to the first token of a 215-token prompt, lower is better</p>

<table class="perf-table">
<thead><tr><th scope="col">Model</th><th scope="col">Metric</th><th scope="col">1 chip</th><th scope="col">2 chips</th><th scope="col">4 chips</th></tr></thead>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3-8B-FP8">Qwen3-8B-FP8</a><span class="perf-arch">dense</span></th><td class="perf-metric">decode</td><td>44.0</td><td>73.1</td><td>112.0</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td>42.3</td><td>68.6</td><td>101.7</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td>251</td><td>166</td><td>123</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3-14B-FP8">Qwen3-14B-FP8</a><span class="perf-arch">dense</span></th><td class="perf-metric">decode</td><td>25.7</td><td>45.8</td><td>74.1</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td>25.1</td><td>44.0</td><td>69.6</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td>394</td><td>237</td><td>168</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3-32B-FP8">Qwen3-32B-FP8</a><span class="perf-arch">dense</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td>22.2</td><td>37.8</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td class="perf-none">—</td><td>21.8</td><td>36.5</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td>480</td><td>319</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3.5-9B">Qwen3.5-9B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">decode</td><td>37.2</td><td>59.3</td><td>85.7</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td>35.2</td><td>54.5</td><td>76.0</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td>458</td><td>286</td><td>197</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3.8-27B-FP8">Qwen3.8-27B-FP8</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td>25.1</td><td>41.5</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td class="perf-none">—</td><td>24.2</td><td>39.1</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td>701</td><td>473</td></tr>
</tbody>
</table>

A dash means the model doesn't fit on one chip. For a single request, SGLang-TorchTPU decodes 1 to 27% slower than SGLang-JAX: least for Qwen3-32B on two chips, most for Qwen3.5-9B on four. Its first tokens take 2.1 to 2.8 times as long. With 16 requests in flight, its total throughput is 47 to 72% of SGLang-JAX's (see [below](#batched-throughput-with-sglang-torchtpu)).

Measured on the same QuietBox 2 with libtt 0.2.0, torch_tpu 0.2.0 and SGLang-TorchTPU `5191118`, from the `main` branch of [sgl-project/sglang-torchtpu](https://github.com/sgl-project/sglang-torchtpu), installed as the [PyTorch SGLang guide](@/docs/pytorch-sglang.md#install) describes. The server ran the [FP8 launch command](@/docs/pytorch-sglang.md#fp8-weights) with `--tp-size` set to the chip count, `--mem-fraction-static 0.85`, `--max-running-requests 2`, `--tpu-compile-tokens 64 256 --chunked-prefill-size 256` and `--disable-radix-cache`. The [hybrid models](@/docs/pytorch-sglang.md#hybrid-models) also need `--no-enable-multimodal --max-mamba-cache-size 8` and `SGLANG_TPU_VIT_PRECOMPILE_SIZES=` (empty). As above, each number decodes 128 tokens, one request at a time, taking the median of five runs after two warmups.

### Sampling with SGLang-TorchTPU

SGLang-TorchTPU filters top-p and top-k without sorting: it searches for each request's cutoff in 30 rounds over the vocabulary, and a `top_k` of at most 64 picks from the 64 most likely tokens. Qwen3-8B-FP8, one request, tokens per second:

| Sampling parameters | 1 chip | 4 chips |
| --- | ---: | ---: |
| greedy (`temperature: 0`) | 44.0 | 112.0 |
| `temperature: 0.7` | 42.3 | 101.7 |
| `temperature: 0.7, top_p: 0.95` | 38.7 | 83.4 |
| `temperature: 0.7, top_k: 20` | 43.0 | 105.6 |
| `temperature: 0.7, top_p: 0.95, top_k: 20` | 42.0 | 100.0 |

These speeds come from libtt 0.2.0's [unrolled short loops](https://github.com/pcmoritz/libtt/pull/520), [full-tile reductions over a few vocabulary rows](https://github.com/pcmoritz/libtt/pull/521) and [traced long eager ops](https://github.com/pcmoritz/libtt/pull/522); before them, `top_p: 0.95` decoded 26.6 tokens/s on one chip and 36.8 on four. Sampling follows the model's probabilities: for each of these settings, the first tokens of 400 requests fit the log-probabilities the server reports.

### Batched throughput with SGLang-TorchTPU

Decode throughput on four chips with several requests in flight, as [above](#batched-throughput): each request sends a 19-token code prompt and generates 128 greedy tokens.

<p class="perf-key"><span class="perf-key-total">total</span> tokens per second across all requests, from sending them until the last one finishes, prefills included<br><span class="perf-key-decode">per user</span> each request's decode rate after its first token, the median over requests</p>

<table class="perf-table">
<thead><tr><th scope="col">Model</th><th scope="col">Metric</th><th scope="col">1 request</th><th scope="col">4 requests</th><th scope="col">8 requests</th><th scope="col">16 requests</th></tr></thead>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-8B-FP8">Qwen3-8B-FP8</a><span class="perf-arch">dense</span></th><td class="perf-metric">total</td><td>103</td><td>352</td><td>567</td><td>741</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>111.4</td><td>100.6</td><td>85.6</td><td>59.1</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-14B-FP8">Qwen3-14B-FP8</a><span class="perf-arch">dense</span></th><td class="perf-metric">total</td><td>69</td><td>241</td><td>413</td><td>599</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>73.9</td><td>68.4</td><td>60.5</td><td>46.1</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-32B-FP8">Qwen3-32B-FP8</a><span class="perf-arch">dense</span></th><td class="perf-metric">total</td><td>35</td><td>126</td><td>215</td><td>333</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>37.7</td><td>35.7</td><td>31.8</td><td>25.9</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.5-9B">Qwen3.5-9B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">total</td><td>76</td><td>238</td><td>355</td><td>455</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>85.6</td><td>68.5</td><td>56.7</td><td>38.2</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.8-27B-FP8">Qwen3.8-27B-FP8</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">total</td><td>36</td><td>113</td><td>169</td><td>230</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>41.5</td><td>37.3</td><td>29.6</td><td>20.4</td></tr>
</tbody>
</table>

The server ran with `--max-running-requests 16 --tpu-compile-bs 1 4 8 16 --torch-compile-max-bs 16`, and the hybrid models with `--max-mamba-cache-size 32`.

## Speculative decoding

### MTP

Qwen3.5 and Qwen3.8 checkpoints include a multi-token prediction (MTP) layer, which SGLang-JAX can run as the draft model: each step, the MTP layer proposes five tokens and the model checks them in one forward pass. Here is Qwen3.8-27B with its own MTP layer as the draft, one request at a time, in tokens per second:

| Request | Chips | Without draft | With draft | Speedup |
| --- | ---: | ---: | ---: | ---: |
| greedy, 5-token prompt | 4 | 46.0 | 119.7 | 2.6× |
| greedy, 19-token code request | 4 | 45.9 | 102.5 | 2.2× |
| greedy, 215-token summarization request | 4 | 45.6 | 56.8 | 1.2× |
| temperature 0.7, 5-token prompt | 4 | 42.8 | 100.2 | 2.3× |
| greedy, 5-token prompt | 2 | 26.6 | 71.8 | 2.7× |
| greedy, 19-token code request | 2 | 26.6 | 61.6 | 2.3× |
| greedy, 215-token summarization request | 2 | 26.6 | 38.5 | 1.5× |
| temperature 0.7, 5-token prompt | 2 | 25.5 | 53.3 | 2.1× |

The greedy text is the same with and without the draft, except for what the model writes on two chips after it has finished answering the summarization request (the requests ignore the end-of-sequence token). The speedup depends on how many of the proposed tokens the model accepts, which varies with the text: it is smallest on the summarization request. Sampling at temperature 0.7 keeps most of it.

SGLang-JAX supports it since [pull request #9](https://github.com/relativity-project/sglang-jax/pull/9). To turn it on, add these flags to the launch command and remove `--disable-overlap-schedule`:

```sh
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path Qwen/Qwen3.8-27B \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --grammar-backend none \
  --max-total-tokens 4096
```

The draft model path is the model itself: SGLang-JAX loads its MTP layer and shares the model's embedding and LM head. We measured the numbers above with overlap scheduling, `JAX_DEFAULT_PRNG_IMPL=unsafe_rbg`, libtt 0.2.0 and SGLang-JAX `6282ec6`, without the draft as well as with it.

SGLang-TorchTPU runs the MTP layer as the draft too, with 3 draft tokens a step. The same requests, in tokens per second:

| Request | Chips | Without draft | With draft | Speedup |
| --- | ---: | ---: | ---: | ---: |
| greedy, 5-token prompt | 4 | 41.5 | 81.5 | 2.0× |
| greedy, 19-token code request | 4 | 41.5 | 81.5 | 2.0× |
| greedy, 215-token summarization request | 4 | 41.2 | 71.9 | 1.7× |
| temperature 0.7, 5-token prompt | 4 | 39.1 | 53.1 | 1.4× |
| greedy, 5-token prompt | 2 | 25.1 | 49.6 | 2.0× |
| greedy, 19-token code request | 2 | 25.1 | 48.4 | 1.9× |
| greedy, 215-token summarization request | 2 | 25.0 | 45.9 | 1.8× |
| temperature 0.7, 5-token prompt | 2 | 24.2 | 41.8 | 1.7× |

As with SGLang-JAX, the model accepts about 3 tokens per step. The greedy text with the draft is the same as without it until a near-tie between two tokens goes the other way, after 120 to 460 characters (the 5-token prompt's text on two chips is the same throughout). Sampling at temperature 0.7 with `top_p: 0.95` and `top_k: 20` as well decodes 48.3 tokens/s on four chips and 38.6 on two. To turn it on, add these flags to the [SGLang-TorchTPU launch command](@/docs/pytorch-sglang.md#hybrid-models):

```sh
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

SGLang-TorchTPU doesn't need a separate draft model path: it loads the checkpoint's MTP layer. We measured with the same SGLang-TorchTPU and libtt as [above](#sglang-torchtpu).

### DFlash

[DFlash](https://github.com/relativity-project/sglang-jax/blob/main/docs/features/speculative_decoding.md) speculative decoding works on one chip with Qwen3 models. Here is Qwen3-8B with the [z-lab/Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) draft model, in tokens per second:

| Prompt | Without draft | With draft | Speedup |
| --- | ---: | ---: | ---: |
| 19-token code request | 43.8 | 124.4 | 2.8× |
| 215-token summarization request | 43.0 | 60.8 | 1.4× |

To turn it on, add these flags to the launch command:

```sh
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --grammar-backend none \
  --max-total-tokens 4096
```

Keep `--disable-overlap-schedule` from the recipe with these flags: SGLang-JAX only runs DFlash with overlap scheduling when `--speculative-num-draft-tokens` is one more than `--speculative-num-steps`. We measured the numbers above with `--disable-overlap-schedule`, libtt 0.2.0 and the same SGLang-JAX as above.

SGLang-TorchTPU runs DFlash on one chip too. Qwen3-8B-FP8 with the same draft, in tokens per second:

| Prompt | Without draft | With draft | Speedup |
| --- | ---: | ---: | ---: |
| 19-token code request | 43.9 | 92.4 | 2.1× |
| 215-token summarization request | 43.0 | 54.3 | 1.3× |

The draft model accepts 3.1 tokens per step. The greedy text is the same as without the draft until a near-tie goes the other way, after 193 and 200 characters. Add `--speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 --max-total-tokens 4096` to the SGLang-TorchTPU launch command.

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

## How we measure

[serving_bench.py](../examples/serving_bench.py) takes this page's measurements against a running server, SGLang-JAX or SGLang-TorchTPU, launched as described above. It needs only Python's standard library:

```sh
curl -fLO https://relativity.cc/examples/serving_bench.py
python3 serving_bench.py single --sampled   # one request at a time
python3 serving_bench.py batched            # 1, 4, 8 and 16 requests at once
```

- **Requests:** every request streams its output and generates exactly 128 tokens, greedy (`temperature: 0`) unless it samples, with `ignore_eos` so that it doesn't stop early. The prompts are `The capital of France is` (5 tokens), a 19-token code request and a 215-token summarization request; the script has their text.
- **Single requests:** each prompt runs twice to warm up and then five times, and the tables show the medians. Decode is the 5-token prompt's tokens after the first, divided by the time from its first streamed token to its last. First token is the time from sending the 215-token prompt to its first streamed token, so it includes the prefill. Sampled decode uses the 5-token prompt. [MTP](#mtp) and [DFlash](#dflash) use the same measurement with the code and summarization requests as well.
- **Batched:** the script sends n copies of the code request at once. Total throughput is the requests' 128 × n tokens divided by the time from sending them until the last one finishes, so it includes the prefills and any wait for a free slot. Per user is the median of the requests' decode rates, each measured from the request's own first token, as for single requests. Total is therefore lower than n times per user; at 16 requests, the prefills take a larger share of the time. Each request count first runs two warmup batches, and more until two in a row return every first token within two seconds, since a server compiles programs for new batch shapes when it first sees them. The tables show the median of the three batches after that.

Each configuration gets a fresh server on chips that were just reset.

## Known gaps

- DFlash doesn't compile on more than one chip yet, and doesn't support Qwen3-14B, Qwen3.5 or Qwen3.8 drafts.
- SGLang-JAX doesn't run the MTP layers of mixture-of-experts checkpoints as drafts yet.
- To use a single chip of a QuietBox, you need a single-chip mesh descriptor in `TT_MESH_GRAPH_DESC_PATH`.
- Qwen3.8-Flash-Next's first request after a start takes about two minutes, while its weights are read from the checkpoint and prepared on the chips.
- Qwen3.8-Flash-Next's QSA layers attend densely, which matches the model only while a request's context is at most 2,051 tokens.
