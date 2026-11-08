+++
title = "Models"
description = "Qwen3, Qwen3.5 and Qwen3.8 models, dense, hybrid and mixture-of-experts, served with SGLang-JAX or SGLang-TorchTPU and libtt on Blackhole, with performance numbers, speculative decoding, and known gaps."
template = "models.html"
+++

## Supported models

These models run on SGLang-JAX's TT backend and libtt. Matmul weights, including the LM head, are stored in BFP8, Tenstorrent's 8-bit block floating point format, in which each group of 16 values shares an exponent. The mixture-of-experts models' experts are the exception: TTNN's fused MoE kernel stores them in BFP4, with 4 bits per value. Everything else (embeddings, norm weights, activations and the KV cache) is BF16. The dense and hybrid models also run on [SGLang-TorchTPU](#sglang-torchtpu), SGLang's PyTorch server, with FP8 weights.

<p class="perf-key"><span class="perf-key-decode">decode</span> tokens per second for a single greedy request without speculative decoding, higher is better<br><span class="perf-key-sampled">sampled</span> the same request sampled at temperature 0.7, without top-p or top-k (see <a href="#sampling">Sampling</a>)<br><span class="perf-key-ttft">first token</span> milliseconds to the first token of a 215-token prompt, lower is better</p>

<table class="perf-table">
<thead><tr><th scope="col">Model</th><th scope="col">Metric</th><th scope="col">1 chip</th><th scope="col">2 chips</th><th scope="col">4 chips</th></tr></thead>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3-8B">Qwen3-8B</a><span class="perf-arch">dense</span></th><td class="perf-metric">decode</td><td>46.0</td><td>80.0</td><td>122.2</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td>44.2</td><td>74.2</td><td>108.0</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td>115</td><td>77</td><td>55</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3-14B">Qwen3-14B</a><span class="perf-arch">dense</span></th><td class="perf-metric">decode</td><td>26.6</td><td>49.0</td><td>83.7</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td>26.0</td><td>46.7</td><td>76.7</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td>182</td><td>112</td><td>76</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3-32B">Qwen3-32B</a><span class="perf-arch">dense</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td>22.5</td><td>40.7</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td class="perf-none">—</td><td>22.0</td><td>39.0</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td>232</td><td>148</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3.5-9B">Qwen3.5-9B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">decode</td><td>40.8</td><td>67.8</td><td>104.7</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td>38.7</td><td>60.9</td><td>88.1</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td>167</td><td>114</td><td>83</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3.8-27B">Qwen3.8-27B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td>24.9</td><td>42.2</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td class="perf-none">—</td><td>23.9</td><td>39.1</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td>286</td><td>194</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3-30B-A3B">Qwen3-30B-A3B</a><span class="perf-arch">MoE</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td>103.7</td><td>126.5</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td class="perf-none">—</td><td>94.1</td><td>111.3</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td>98</td><td>72</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3.5-35B-A3B">Qwen3.5-35B-A3B</a><span class="perf-arch">MoE</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td>77.2</td><td>95.6</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td class="perf-none">—</td><td>68.4</td><td>81.6</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td>149</td><td>110</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Qwen3.8-Flash-Next</a><span class="perf-arch">MoE</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td class="perf-none">—</td><td>40.0</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td class="perf-none">—</td><td class="perf-none">—</td><td>36.7</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td class="perf-none">—</td><td>815</td></tr>
</tbody>
</table>

A dash means the model doesn't fit on one chip or we haven't run that configuration yet. Dense models use full attention in every layer; the hybrid models (Qwen3.5, Qwen3.8) mix gated DeltaNet layers with full attention. The mixture-of-experts (MoE) models run each token through a few of their experts, with the experts split over the chips. Qwen3-30B-A3B and Qwen3.5-35B-A3B use 8 of 128 and 256 experts, about 3B active parameters, and need at least two chips; Qwen3.8-Flash-Next uses 10 of 512 and needs four. Qwen3.5-35B-A3B and Qwen3.8-Flash-Next are also hybrid.

Measured on a [QuietBox 2](../hardware/#quietbox-2) with libtt 0.1.5 and SGLang-JAX `6282ec6`, using the [inference recipe](@/docs/inference.md) with `--tp-size` set to the chip count, overlap scheduling on (remove `--disable-overlap-schedule` from the launch command) and `JAX_DEFAULT_PRNG_IMPL=unsafe_rbg` (see [Sampling](#sampling)). Each number decodes 128 tokens, one request at a time and without [speculative decoding](#speculative-decoding), taking the median of five runs after two warmups.

## Sampling

Set `JAX_DEFAULT_PRNG_IMPL=unsafe_rbg` in the launch command's environment when requests sample:

```sh
export JAX_DEFAULT_PRNG_IMPL=unsafe_rbg
```

It makes JAX draw its random bits with libtt's native generator. JAX's default Threefry generator adds about 18 ms to every sampled token on Blackhole for the Qwen3 models' 151,936-token vocabulary, and 29 ms for Qwen3.8-Flash-Next's 248,320. That slows sampled decode on one chip from 44.2 to about 24 tokens per second for Qwen3-8B.

With it, sampling at temperature 0.7 adds 0.9 to 1.1 ms per token for the Qwen3 models and 1.3 to 2.2 ms for Qwen3.5 and Qwen3.8, whose vocabulary is larger. Its share grows as decode gets faster: from 2% for Qwen3-32B on two chips to 16% for Qwen3.5-9B on four. `top_p` and `top_k` each add a search over the vocabulary to every step, and a request that sets only `temperature` runs neither. Qwen3-8B, one request, tokens per second:

| Sampling parameters | 1 chip | 4 chips |
| --- | ---: | ---: |
| greedy (`temperature: 0`) | 46.0 | 122.2 |
| `temperature: 0.7` | 44.2 | 108.0 |
| `temperature: 0.7, top_p: 0.95` | 38.6 | 79.9 |
| `temperature: 0.7, top_k: 20` | 38.7 | 80.1 |
| `temperature: 0.7, top_p: 0.95, top_k: 20` | 34.3 | 63.3 |

In a batch, a search runs for every request as long as any request in the batch needs it. `min_p` filters without a search.

## Batched throughput

Decode throughput on four chips with several requests in flight. Each request sends a 19-token code prompt and generates 128 tokens.

<p class="perf-key"><span class="perf-key-total">total</span> tokens per second across all running requests<br><span class="perf-key-decode">per user</span> tokens per second for each request</p>

<table class="perf-table">
<thead><tr><th scope="col">Model</th><th scope="col">Metric</th><th scope="col">1 request</th><th scope="col">4 requests</th><th scope="col">8 requests</th><th scope="col">16 requests</th></tr></thead>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-8B">Qwen3-8B</a><span class="perf-arch">dense</span></th><td class="perf-metric">total</td><td>118</td><td>402</td><td>723</td><td>1101</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>121.8</td><td>104.7</td><td>98.3</td><td>79.2</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-14B">Qwen3-14B</a><span class="perf-arch">dense</span></th><td class="perf-metric">total</td><td>82</td><td>286</td><td>522</td><td>870</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>83.5</td><td>75.0</td><td>70.2</td><td>60.9</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-32B">Qwen3-32B</a><span class="perf-arch">dense</span></th><td class="perf-metric">total</td><td>40</td><td>141</td><td>266</td><td>467</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>40.6</td><td>37.2</td><td>36.0</td><td>32.5</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.5-9B">Qwen3.5-9B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">total</td><td>101</td><td>345</td><td>578</td><td>897</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>104.6</td><td>91.3</td><td>77.7</td><td>63.1</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.8-27B">Qwen3.8-27B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">total</td><td>41</td><td>143</td><td>252</td><td>405</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>42.1</td><td>37.5</td><td>33.9</td><td>27.7</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-30B-A3B">Qwen3-30B-A3B</a><span class="perf-arch">MoE</span></th><td class="perf-metric">total</td><td>121</td><td>388</td><td>670</td><td>973</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>126.2</td><td>104.3</td><td>93.5</td><td>71.8</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.5-35B-A3B">Qwen3.5-35B-A3B</a><span class="perf-arch">MoE</span></th><td class="perf-metric">total</td><td>92</td><td>303</td><td>490</td><td>720</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>95.4</td><td>80.8</td><td>66.5</td><td>51.0</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Qwen3.8-Flash-Next</a><span class="perf-arch">MoE</span></th><td class="perf-metric">total</td><td>37</td><td>95</td><td>135</td><td>162</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>39.4</td><td>26.2</td><td>19.8</td><td>11.1</td></tr>
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
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3-8B-FP8">Qwen3-8B-FP8</a><span class="perf-arch">dense</span></th><td class="perf-metric">decode</td><td>44.0</td><td>69.3</td><td>103.9</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td>42.3</td><td>65.2</td><td>94.9</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td>251</td><td>168</td><td>125</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3-14B-FP8">Qwen3-14B-FP8</a><span class="perf-arch">dense</span></th><td class="perf-metric">decode</td><td>25.7</td><td>44.3</td><td>70.3</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td>25.1</td><td>42.6</td><td>66.2</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td>393</td><td>238</td><td>170</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3-32B-FP8">Qwen3-32B-FP8</a><span class="perf-arch">dense</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td>21.9</td><td>36.8</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td class="perf-none">—</td><td>21.4</td><td>35.6</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td>480</td><td>320</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3.5-9B">Qwen3.5-9B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">decode</td><td>37.2</td><td>59.3</td><td>85.7</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td>35.3</td><td>54.5</td><td>76.0</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td>480</td><td>296</td><td>204</td></tr>
</tbody>
<tbody>
<tr class="perf-decode"><th scope="rowgroup" rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3.8-27B-FP8">Qwen3.8-27B-FP8</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">decode</td><td class="perf-none">—</td><td>25.1</td><td>41.5</td></tr>
<tr class="perf-sampled"><td class="perf-metric">sampled</td><td class="perf-none">—</td><td>24.2</td><td>39.1</td></tr>
<tr class="perf-ttft"><td class="perf-metric">first token</td><td class="perf-none">—</td><td>734</td><td>488</td></tr>
</tbody>
</table>

A dash means the model doesn't fit on one chip. For a single request, SGLang-TorchTPU decodes up to 18% slower than SGLang-JAX (Qwen3.5-9B on four chips), and as fast for Qwen3.8-27B. Its first tokens take 2.1 to 2.9 times as long. With 16 requests in flight, its total throughput is 51 to 72% of SGLang-JAX's (see [below](#batched-throughput-with-sglang-torchtpu)).

Measured on the same QuietBox 2 with SGLang-TorchTPU `5191118`, the `main` branch of [sgl-project/sglang-torchtpu](https://github.com/sgl-project/sglang-torchtpu), and libtt `main` (`5daedd1`) with three changes for sampling that aren't merged yet (branch [`trace-long-eager-programs`](https://github.com/pcmoritz/libtt/tree/trace-long-eager-programs), see [Sampling with SGLang-TorchTPU](#sampling-with-sglang-torchtpu)). The server ran the [FP8 launch command](@/docs/pytorch-sglang.md#fp8-weights) with `--tp-size` set to the chip count, `--mem-fraction-static 0.85`, `--max-running-requests 2`, `--tpu-compile-tokens 64 256 --chunked-prefill-size 256` and `--disable-radix-cache`. The [hybrid models](@/docs/pytorch-sglang.md#hybrid-models) also need `--no-enable-multimodal --max-mamba-cache-size 8` and `SGLANG_TPU_VIT_PRECOMPILE_SIZES=` (empty). As above, each number decodes 128 tokens, one request at a time, taking the median of five runs after two warmups.

### Sampling with SGLang-TorchTPU

SGLang-TorchTPU filters top-p and top-k without sorting: it searches for each request's cutoff in 30 rounds over the vocabulary, and a `top_k` of at most 64 picks from the 64 most likely tokens. Qwen3-8B-FP8, one request, tokens per second:

| Sampling parameters | 1 chip | 4 chips |
| --- | ---: | ---: |
| greedy (`temperature: 0`) | 44.0 | 103.9 |
| `temperature: 0.7` | 42.3 | 94.9 |
| `temperature: 0.7, top_p: 0.95` | 38.7 | 78.5 |
| `temperature: 0.7, top_k: 20` | 43.0 | 98.7 |
| `temperature: 0.7, top_p: 0.95, top_k: 20` | 42.0 | 93.5 |

These need libtt's [unrolled short loops](https://github.com/pcmoritz/libtt/compare/main...unroll-counted-loops), [full-tile reductions over a few vocabulary rows](https://github.com/pcmoritz/libtt/compare/unroll-counted-loops...wide-rows-full-tiles) and [traced long eager ops](https://github.com/pcmoritz/libtt/compare/wide-rows-full-tiles...trace-long-eager-programs): on libtt `main`, `top_p: 0.95` decodes 26.6 tokens/s on one chip and 36.8 on four. Sampling follows the model's probabilities: for each of these settings, the first tokens of 400 requests fit the log-probabilities the server reports.

### Batched throughput with SGLang-TorchTPU

Decode throughput on four chips with several requests in flight, as [above](#batched-throughput): each request sends a 19-token code prompt and generates 128 greedy tokens.

<p class="perf-key"><span class="perf-key-total">total</span> tokens per second across all running requests<br><span class="perf-key-decode">per user</span> tokens per second for each request</p>

<table class="perf-table">
<thead><tr><th scope="col">Model</th><th scope="col">Metric</th><th scope="col">1 request</th><th scope="col">4 requests</th><th scope="col">8 requests</th><th scope="col">16 requests</th></tr></thead>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-8B-FP8">Qwen3-8B-FP8</a><span class="perf-arch">dense</span></th><td class="perf-metric">total</td><td>96</td><td>342</td><td>555</td><td>734</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>103.5</td><td>97.3</td><td>83.1</td><td>59.8</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-14B-FP8">Qwen3-14B-FP8</a><span class="perf-arch">dense</span></th><td class="perf-metric">total</td><td>66</td><td>236</td><td>406</td><td>593</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>70.1</td><td>66.8</td><td>59.3</td><td>46.1</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-32B-FP8">Qwen3-32B-FP8</a><span class="perf-arch">dense</span></th><td class="perf-metric">total</td><td>35</td><td>124</td><td>213</td><td>335</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>36.7</td><td>35.2</td><td>31.4</td><td>25.8</td></tr>
</tbody>
<tbody>
<tr class="perf-total"><th scope="rowgroup" rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.5-9B">Qwen3.5-9B</a><span class="perf-arch">hybrid</span></th><td class="perf-metric">total</td><td>76</td><td>198</td><td>354</td><td>454</td></tr>
<tr class="perf-decode"><td class="perf-metric">per user</td><td>85.7</td><td>71.8</td><td>56.6</td><td>38.8</td></tr>
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
| greedy, 5-token prompt | 4 | 42.2 | 106.4 | 2.5× |
| greedy, 19-token code request | 4 | 42.1 | 86.6 | 2.1× |
| greedy, 215-token summarization request | 4 | 41.9 | 58.4 | 1.4× |
| temperature 0.7, 5-token prompt | 4 | 39.1 | 90.1 | 2.3× |
| greedy, 5-token prompt | 2 | 24.9 | 65.3 | 2.6× |
| greedy, 19-token code request | 2 | 24.9 | 67.4 | 2.7× |
| greedy, 215-token summarization request | 2 | 24.9 | 55.9 | 2.2× |
| temperature 0.7, 5-token prompt | 2 | 24.0 | 50.1 | 2.1× |

The greedy text is the same with and without the draft, except for the last few tokens of the summarization request on two chips, after the model had finished its answer. The speedup depends on how many of the proposed tokens the model accepts, which varies with the text: it is smallest on the summarization request. Sampling at temperature 0.7 keeps most of it.

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

The draft model path is the model itself: SGLang-JAX loads its MTP layer and shares the model's embedding and LM head. We measured the numbers above with overlap scheduling, `JAX_DEFAULT_PRNG_IMPL=unsafe_rbg`, libtt 0.1.5 and SGLang-JAX `6282ec6`, without the draft as well as with it.

SGLang-TorchTPU runs the MTP layer as the draft too, with 3 draft tokens a step. The same requests, in tokens per second:

| Request | Chips | Without draft | With draft | Speedup |
| --- | ---: | ---: | ---: | ---: |
| greedy, 5-token prompt | 4 | 41.5 | 62.5 | 1.5× |
| greedy, 19-token code request | 4 | 41.5 | 58.1 | 1.4× |
| greedy, 215-token summarization request | 4 | 41.2 | 56.4 | 1.4× |
| temperature 0.7, 5-token prompt | 4 | 39.1 | 49.1 | 1.3× |
| greedy, 5-token prompt | 2 | 25.1 | 43.9 | 1.8× |
| greedy, 19-token code request | 2 | 25.1 | 39.7 | 1.6× |
| greedy, 215-token summarization request | 2 | 25.0 | 37.2 | 1.5× |
| temperature 0.7, 5-token prompt | 2 | 24.2 | 34.3 | 1.4× |

The model accepts about 3 tokens per step. As with SGLang-JAX, the greedy text with the draft is the same as without it until a near-tie between two tokens goes the other way, after 180 to 420 characters (the 5-token prompt's text on two chips is the same throughout). Sampling at temperature 0.7 with `top_p: 0.95` and `top_k: 20` as well decodes 47.9 tokens/s on four chips and 29.9 on two. To turn it on, add these flags to the [SGLang-TorchTPU launch command](@/docs/pytorch-sglang.md#hybrid-models):

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
| 19-token code request | 44.0 | 126.7 | 2.9× |
| 215-token summarization request | 43.1 | 75.8 | 1.8× |

To turn it on, add these flags to the launch command:

```sh
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --grammar-backend none \
  --max-total-tokens 4096
```

Keep `--disable-overlap-schedule` from the recipe with these flags: SGLang-JAX only runs DFlash with overlap scheduling when `--speculative-num-draft-tokens` is one more than `--speculative-num-steps`. We measured the numbers above with `--disable-overlap-schedule`, libtt 0.1.5 and the same SGLang-JAX as above.

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
- SGLang-TorchTPU doesn't run DFlash on libtt yet: its draft writes the draft's KV cache with a Pallas kernel that libtt doesn't replace yet.
- SGLang-JAX doesn't run the MTP layers of mixture-of-experts checkpoints as drafts yet.
- To use a single chip of a QuietBox, you need a single-chip mesh descriptor in `TT_MESH_GRAPH_DESC_PATH`.
- Qwen3.8-Flash-Next's first request after a start takes about two minutes, while its weights are read from the checkpoint and prepared on the chips.
- Qwen3.8-Flash-Next's QSA layers attend densely, which matches the model only while a request's context is at most 2,051 tokens.
