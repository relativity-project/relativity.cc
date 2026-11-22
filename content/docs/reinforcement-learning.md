+++
title = "Reinforcement learning"
description = "Train Qwen3-0.6B on GSM8K with GRPO in miles on a QuietBox 2: SGLang-TorchTPU generates on two chips, miles' torchtitan trainer trains on the other two, and the weights move between them through a directory. Experimental."
weight = 45
+++

[miles](https://github.com/radixark/miles) is RadixArk's framework for reinforcement learning on large language models. It generates rollouts with SGLang and trains with Megatron-LM, FSDP or torchtitan, and on TPUs it runs the torchtitan trainer and [SGLang-TorchTPU](https://github.com/sgl-project/sglang-torchtpu) through [torch_tpu](https://github.com/google-pytorch/torch_tpu). libtt implements the PJRT C API that torch_tpu drives, so the same path runs on Blackhole chips. This guide trains Qwen3-0.6B to solve grade-school math problems with GRPO on the four chips of a [QuietBox 2](../../hardware/#quietbox-2).

This path is **experimental**: it needs libtt built from `main` and a small change to miles, it has only run Qwen3-0.6B, and it is slow (see [Speed](#speed)).

After 60 GRPO steps of 128 answers each, Qwen3-0.6B answers 72.4% of the GSM8K test questions, up from 57.9% (all 1,319 questions, greedy decoding, the number in the last `\boxed{}` compared with the answer):

| | Before training* | After 30 steps | After 60 steps | After 90 steps |
| --- | --- | --- | --- | --- |
| GSM8K test accuracy | 57.9% | 70.7% | **72.4%** | **72.4%** |
| Mean answer length | 186 tokens | 348 tokens | 359 tokens | 366 tokens |
| Answers cut off at the limit | 1.7% | 3.9% | 4.4% | 7.8% |

\* Measured with answers cut off at 512 tokens instead of 1,024, in an earlier run of the same model.

The training reward, the fraction of sampled answers that are right, averaged over 15 rollouts at a time, kept rising after the test accuracy stopped:

| Rollouts | 1–15 | 16–30 | 31–45 | 46–60 | 61–75 | 76–90 |
| --- | --- | --- | --- | --- | --- | --- |
| Reward | 0.64 | 0.68 | 0.70 | 0.72 | 0.75 | 0.77 |

The model learned to write longer solutions, and by the end more of them ran into the 1,024-token limit, which scores them 0. A first run that cut answers off at 512 tokens reached 66.5% after 30 steps, against 70.7% here. The 90-step run took 3 hours 48 minutes, 48 of them for the three evaluations after training started.

## How it fits together

Each rollout starts from 16 GSM8K training questions and samples 8 answers to each. miles' math reward gives an answer 1 when the number in its last `\boxed{}` matches the question's, and GRPO scores each answer against the others for its question. The trainer takes one optimizer step on the 128 answers and sends the new weights to the engine, which generates the next rollout with them.

| Chips | Job | Software |
| --- | --- | --- |
| 0 and 1 | Training | miles' torchtitan trainer, one process per chip, the model sharded across both with FSDP2 |
| 2 and 3 | Generation | SGLang-TorchTPU, tensor parallel over both |

On a TPU host, miles moves the weights from trainer to engine chip to chip, which needs the two jobs to share one TPU slice. Here each job has its own chips, and the weights go through a directory instead: miles' `disk-delta` mode compares the trained weights with the previous version, writes the bytes that changed, compressed, and the engine patches its own copy of the checkpoint and reloads it. A step changes about 0.5% of Qwen3-0.6B's bytes, so each update writes 20 MB and takes 1.5 seconds.

libtt also fills in for the parts of the TPU path that only run on TPUs:

- miles' trainer runs attention with a Pallas splash attention kernel. libtt replaces it with tt-train's fused attention ([libtt #559](https://github.com/pcmoritz/libtt/pull/559)), as it does for [TorchTitan's TPU trainer](https://github.com/google-pytorch/torchtitan/tree/main/torchtitan/experiments/tpu).
- PyTorch's FSDP2 gathers parameters into flat buffers and reduces gradients from `[2, N]` ones, and miles computes log-probabilities on `[tokens, 1, vocabulary]` logits. A Blackhole chip stores tensors in 32 × 32 tiles of their last two dimensions, so each of these would take 16 to 32 times its size: the gathered embedding alone 10 GB. libtt stores them in whole tiles instead (`[N / 32, 32]`, and `[tokens, vocabulary]`), the way a TPU picks a layout without the padding ([libtt #566](https://github.com/pcmoritz/libtt/pull/566)).

## Install

You need a QuietBox 2 (or another machine with four Blackhole chips) with working drivers, [uv](https://docs.astral.sh/uv/getting-started/installation/), Git and [Bazelisk](https://github.com/bazelbuild/bazelisk). Run the [device check](@/docs/getting-started.md#verify-jax-execution) first.

Work in one directory: the steps below put SGLang-TorchTPU, libtt, miles and torchtitan, the model and the data side by side in it.

**1. SGLang-TorchTPU.** Install it as in the [PyTorch SGLang guide](@/docs/pytorch-sglang.md#install), in `sglang-torchtpu/.venv`. It serves the policy, and its packages also give miles' trainer the SGLang server arguments and TPU platform it imports.

**2. libtt.** The miles path needs these two libtt changes, which came after the 0.2.1 release, so build the plugin wheel from `main` as in [Build and test libtt](@/docs/software-stack.md#build-and-test-libtt) and install it in SGLang-TorchTPU's environment, in place of the PyPI release:

```sh
git clone https://github.com/pcmoritz/libtt.git
(cd libtt && bazel build -c opt //:jax_tt_plugin_wheel)
VIRTUAL_ENV=sglang-torchtpu/.venv uv pip install --no-deps --reinstall libtt/bazel-bin/jax_tt_plugin-*.whl
```

**3. miles.** miles' TPU trainer needs PyTorch 2.14 or newer and torchtitan at the commit miles pins, with miles' patch for torch_tpu, so it gets its own environment. Check out miles at the tested commit and apply [one change](../../examples/miles-tpu-disk-delta.patch), which lets its `disk-delta` weight transfer run on TPU platforms (it only allowed chip-to-chip transfers there) and replaces two CUDA calls in it:

```sh
git clone https://github.com/radixark/miles.git
git clone https://github.com/pytorch/torchtitan.git
(cd torchtitan && git checkout 45e33cd4 && patch -p1 --fuzz=0 < ../miles/docker/tpu_patch/torchtitan_torch_tpu.patch)
cd miles
git checkout 7767e15
curl -fLO https://relativity.cc/examples/miles-tpu-disk-delta.patch
git apply miles-tpu-disk-delta.patch
uv venv -p 3.12 .venv
printf "torch==2.14.1+cpu\ntorchvision==0.29.1+cpu\ntorch_tpu==0.2.0\njax==0.11.1\njaxlib==0.11.1\ntriton==3.8.0\n" > constraints.txt
grep -vE '^\s*(#|$)|nvidia-resiliency-ext|ring_flash_attn|torchft-nightly' requirements.txt > requirements-tt.txt
VIRTUAL_ENV=.venv uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu -c constraints.txt \
  torch torchvision torch_tpu jax jaxlib triton -e ../torchtitan -r requirements-tt.txt \
  sentencepiece compressed_tensors gguf partial_json_parser msgspec
VIRTUAL_ENV=.venv uv pip install --no-deps -e . -e ../sglang-torchtpu/python -e ../sglang-torchtpu \
  ../libtt/bazel-bin/jax_tt_plugin-*.whl
cd ..
```

The constraints pin the versions the trainer was tested with: torch_tpu 0.2.0 accepts PyTorch 2.11 to 2.15, and miles' own Docker image uses 2.14.1. `requirements-tt.txt` is miles' requirements without the three that only build for CUDA, as miles' TPU image installs them. The last few packages of the first install are imported by the SGLang modules miles loads.

**4. Model and data.** Download Qwen3-0.6B and miles' copy of GSM8K, whose questions come with a system prompt asking for the answer in `\boxed{}`:

```sh
miles/.venv/bin/hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
miles/.venv/bin/hf download zhuzilin/gsm8k --repo-type dataset --local-dir datasets/gsm8k
```

Use `--local-dir`: SGLang-TorchTPU copies the checkpoint with its file permissions and patches the copy in place with each update, so a read-only file, as Hugging Face cache files can be, fails the first update.

## Run

Download the [launcher](../../examples/miles_gsm8k.sh) and start it in the same directory:

```sh
curl -fLO https://relativity.cc/examples/miles_gsm8k.sh
bash miles_gsm8k.sh
```

It starts SGLang-TorchTPU on chips 2 and 3, waits until it serves, starts a Ray head for the trainer's two processes, and runs miles' `train.py` on chips 0 and 1 with the recipe below. Logs, the published weight updates and the engine's copy of the checkpoint go to `runs/<date>-<time>/`; `train.log` has each rollout's reward and each evaluation's accuracy. Stopping the script (Ctrl-C) stops the engine and Ray too.

| Setting | Value |
| --- | --- |
| Rollouts | 90, each 16 questions with 8 answers sampled at temperature 1, up to 1,024 tokens |
| Algorithm | GRPO, clipping 0.2 / 0.28, no KL or entropy term |
| Optimizer | Adam, learning rate 2e-6, weight decay 0.1, one step per rollout (128 answers) |
| Training | BF16 compute, FP32 parameters and optimizer state, FSDP2 over two chips, up to 2,048 tokens per microbatch, activation checkpointing |
| Evaluation | All 1,319 GSM8K test questions, greedy, before training and every 30 steps |

`ROLLOUTS`, `PROMPTS`, `SAMPLES`, `LR`, `MAX_RESPONSE` and `EVAL_INTERVAL` change the recipe, the variables at the top of the script say where things are, and further arguments go to `train.py`.

## Speed

A rollout and its training step take about two minutes, with answers of about 370 tokens:

| Phase | Time |
| --- | --- |
| Generating 128 answers (SGLang-TorchTPU, two chips) | 47 s |
| Log-probabilities of the answers under the weights that generated them (trainer) | 23 s |
| Training step | 49 s |
| Weight update | 1.5 s |

The trainer processes about 1,120 tokens per second per chip, 2.1 TFLOPS. Generation decodes about 1,080 tokens per second with 60 to 64 answers in flight, and an evaluation of the 1,319 test questions takes 16 minutes. Neither has been profiled yet. The first rollout also waits for the trainer's programs to compile, about a minute.

Gathering FSDP2's parameters once per step instead of once per microbatch (`--titan-fsdp-gather-once`, which miles' own TPU recipe for Qwen3-4B uses) made the training step 2.4 times as fast; at 0.6B parameters, the whole model and its gradients fit on each chip.

## Next steps

- **Upstream the change to miles:** allowing `disk-delta` on TPU platforms isn't specific to Tenstorrent.
- **Speed:** generation and the trainer run well below what the chips do elsewhere (compare [TorchTitan training](https://github.com/google-pytorch/torchtitan/tree/main/torchtitan/experiments/tpu) and the [models page](@/models/_index.md#sglang-torchtpu)).
- **Checkpoints:** saving one (`--save`) fails on Tenstorrent chips: PyTorch's distributed checkpointing gathers each process's save plan as pickled bytes, and an operation on that megabyte-long row of bytes needs more than a core's memory. A run doesn't save its trained weights yet.
- **Answer length:** the test accuracy stopped rising once more answers reached the length limit; a longer limit or a length penalty may move it further.
- **Larger models:** only Qwen3-0.6B has run. Qwen3-4B, which miles' TPU recipe trains, needs about seven times the trainer's memory per chip.
