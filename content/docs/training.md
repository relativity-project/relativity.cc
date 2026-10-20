+++
title = "Training"
description = "Train Qwen3-1.7B on the four chips of a QuietBox 2 with TorchTitan and TorchTPU, launched the same way as on a Cloud TPU."
weight = 40
+++

## Train Qwen3-1.7B on four chips

[TorchTPU](https://github.com/google-pytorch/torch_tpu) is Google's PyTorch backend for TPUs. It drives its chips through the PJRT C API, which libtt implements, so TorchTPU also runs on Blackhole. This guide trains Qwen3-1.7B from scratch on the four chips of a [QuietBox 2](../../hardware/#quietbox-2) with [TorchTitan's TPU trainer](https://github.com/google-pytorch/torchtitan/tree/61bdf4c2/torchtitan/experiments/tpu). TorchTitan and TorchTPU are unmodified, and the install and launch follow their instructions for a TPU VM, with a few additions for the Blackhole chips.

The run uses TorchTitan's `qwen3_1_7b` preset:

| Setting | Value |
| --- | --- |
| Model | Qwen3-1.7B: 28 layers, 2.03 billion parameters with separate input and output embeddings |
| Data | [C4](https://huggingface.co/datasets/allenai/c4) English, streamed from Hugging Face, with the Qwen3 tokenizer |
| Parallelism | FSDP across 4 chips, one process per chip |
| Batch | One 2,048-token sequence per chip, 8,192 tokens per step |
| Precision | BF16 compute, FP32 parameters and optimizer state, BF16 gradient reduction |
| Optimizer | AdamW, learning rate 8e-4, weight decay 0.1, gradient clipping at 1.0 |
| Memory | Full activation checkpointing |
| Compilation | `torch.compile` of each transformer block |

A 500-step run with two checkpoints took 23 minutes. After the first step, which compiles each process's programs, a step takes 2.55 seconds: **804 tokens per second per chip**, 3,220 for the four. Over the 4.1 million tokens, the loss falls like this, averaged over 50 steps:

| Steps | 1–50 | 51–100 | 101–150 | 151–200 | 201–250 | 251–300 | 301–350 | 351–400 | 401–450 | 451–500 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Loss | 8.35 | 7.22 | 6.96 | 6.83 | 6.69 | 6.57 | 6.52 | 6.41 | 6.29 | 6.36 |

The first step's loss is 12.33, a little above the 11.93 of a uniform guess over the 151,936-token vocabulary. Each step sees only 8,192 tokens, so single steps scatter around these averages with a standard deviation of 0.2 to 0.35.

## 1. Install

You need a QuietBox 2 with working drivers (or another machine with four Blackhole chips), [uv](https://docs.astral.sh/uv/getting-started/installation/), Git and [Bazelisk](https://github.com/bazelbuild/bazelisk). Run the [device check](@/docs/getting-started.md#verify-jax-execution) first.

TorchTPU on several chips needs libtt's [`torchtitan-training` branch](https://github.com/pcmoritz/libtt/tree/torchtitan-training), which isn't in a released `jax-tt-plugin` wheel yet. Build the wheel from it, as in [Build and test libtt](@/docs/software-stack.md#build-and-test-libtt). The first build compiles LLVM, the compiler and the runtime, so it takes a while:

```sh
git clone -b torchtitan-training https://github.com/pcmoritz/libtt.git
(cd libtt && bazel build -c opt //:jax_tt_plugin_wheel)
```

Then clone TorchTitan at the tested commit and install its TPU dependencies in the order of [TorchTitan's README](https://github.com/google-pytorch/torchtitan/blob/61bdf4c2/torchtitan/experiments/tpu/README.md). Its files pin conflicting versions, so install them one after another rather than in one command:

```sh
git clone https://github.com/google-pytorch/torchtitan.git
cd torchtitan
git checkout 61bdf4c2
uv venv --python 3.12
source .venv/bin/activate
CPU="--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match"
uv pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
uv pip install -r torchtitan/experiments/tpu/requirements.txt \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html $CPU
uv pip install -r requirements.txt -r .ci/docker/requirements-flux.txt $CPU
uv pip install "torch_tpu==0.2.0" ../libtt/bazel-bin/jax_tt_plugin-0.1.2-py3-none-linux_x86_64.whl $CPU
```

The README installs a PyTorch nightly. TorchTPU 0.2.0 accepts PyTorch 2.11 to 2.15, and this guide uses the 2.14.1 release, which libtt's TorchTPU support is tested with.

Download the Qwen3 tokenizer with TorchTitan's script. The model itself starts from random weights:

```sh
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-1.7B --assets tokenizer
```

## 2. Train

Turn on TorchTPU support in libtt, let TorchTPU describe the chips to `torchrun`, and start one process per chip:

```sh
export LIBTT_TPU_COMPAT=1
eval $(python -m torch.tpu.distributed | sed 's/^/export /')
env -u TT_METAL_RUNTIME_ROOT -u TT_MESH_GRAPH_DESC_PATH -u TT_VISIBLE_DEVICES \
torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train \
  --module=torchtitan.experiments.tpu.qwen3 \
  --config=qwen3_1_7b \
  --splash_attention_kernel.no_use_splash_attention_kernel \
  --hf_assets_path=assets/hf/Qwen3-1.7B \
  --dataloader.dataset=c4 \
  --training.steps=500 \
  --lr_scheduler.warmup_steps=50
```

What each part does:

- `LIBTT_TPU_COMPAT=1` makes TorchTPU load libtt instead of Google's libtpu. It also reports the Blackhole chips as TorchTPU devices and turns off TorchTPU's Pallas flash attention, which only TPUs can run.
- `python -m torch.tpu.distributed` is TorchTPU's launch helper. It prints the chip topology and one address per process, which TorchTPU uses to give each process its own chip and to connect the processes.
- `env -u` clears settings from other recipes, such as the [inference guide's](@/docs/inference.md#use-several-chips) chip list. libtt sets the chip and a four-chip mesh for each process itself.
- `--splash_attention_kernel.no_use_splash_attention_kernel` turns off the preset's splash attention, another Pallas kernel. Attention runs through PyTorch's `scaled_dot_product_attention` instead.
- `--dataloader.dataset=c4` streams C4 instead of the small test file the preset trains on. `--training.steps` and `--lr_scheduler.warmup_steps` shorten the preset's schedule for a run of about half an hour.

Each process compiles its programs during the first step: about 20 seconds when TT-Metal's kernels are already in its cache, and two to three minutes when it has to build them. After that, every process prints a line per step:

```text
step: 300  loss:  6.60610  grad_norm:  0.8004  memory: -0.00GiB(-0.00%)  tps: 805  tflops: 9.44  mfu: 3.03%  time_end_to_end: 2.54s
```

`tps` is tokens per second per chip. TorchTitan doesn't know the Blackhole's peak throughput and computes `mfu` against an A100, and libtt doesn't report device memory to PyTorch, so ignore those two fields.

## 3. Save and resume

TorchTitan saves checkpoints with PyTorch's distributed checkpointing: each process writes its shards of the model and the optimizer state, and the data loader's position. Add `--checkpoint.enable` to the training command to save to `outputs/checkpoint` every 500 steps (`--checkpoint.interval`) and after the last step. Running the command again with a larger `--training.steps` resumes from the latest checkpoint.

With streamed C4, also add `--checkpoint.exclude_from_loading dataloader` when you resume. TorchTitan can't load the data loader's position for a streamed dataset and stops with `Missing key in checkpoint state_dict: dataloader.dp_rank_0.parent_state.parent.hf.examples_iterable.previous_state`, on the CPU as well, so the flag loads the model and the optimizer and starts the data stream over.

A Qwen3-1.7B checkpoint, with the AdamW state, takes 23 GB and 13 to 14 seconds to save. A save also compiles its programs the first time a process makes one, which can add about a minute when TT-Metal has to build their kernels.

Resuming is exact. With TorchTitan's local test data, whose loader position loads fine, a run resumed from its step-3 checkpoint reproduced steps 4 to 6, and its step-6 checkpoint matched the uninterrupted run's bit for bit in all 1,076 tensors. The C4 run above, resumed from step 500 with the flag, loaded its checkpoint in 30 seconds and went on at the same loss, around 6.2, and the same speed.

## 4. Check the numbers against the CPU

A falling loss doesn't prove that the chips compute the right thing, so compare against PyTorch on the CPU. TorchTitan can write a seed checkpoint with a model's initial weights, and runs that load it with `--checkpoint.load_only` train the same model on the same data without saving. With `TORCH_DEVICE_BACKEND_AUTOLOAD=0`, TorchTPU doesn't load and PyTorch runs on the CPU. TorchTitan's debug model and test data keep the CPU run to a few minutes per step:

```sh
ARGS="--module=torchtitan.experiments.tpu.qwen3 --config=qwen3_debugmodel \
  --parallelism.use_simple_fsdp --training.steps=4 \
  --checkpoint.enable --checkpoint.folder=ckpt/seed"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 torchrun --nproc_per_node=1 \
  -m torchtitan.experiments.tpu.train $ARGS --checkpoint.create_seed_checkpoint
TORCH_DEVICE_BACKEND_AUTOLOAD=0 torchrun --nproc_per_node=4 \
  -m torchtitan.experiments.tpu.train $ARGS --checkpoint.load_only
torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train $ARGS --checkpoint.load_only
```

These are the losses and gradient norms of the 4-process CPU run and the 4-chip run:

| Step | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| CPU loss | 7.67712 | 6.75435 | 6.19516 | 5.97025 |
| Chip loss | 7.67713 | 6.75506 | 6.19494 | 5.97012 |
| CPU gradient norm | 1.9657 | 2.9336 | 3.6235 | 3.4442 |
| Chip gradient norm | 1.9618 | 2.9295 | 3.6230 | 3.4439 |

After the four steps, the chips' weights differ from the CPU's by at most 0.15% (relative norm per tensor, 0.09% for the median tensor), while training moved them by about 8% from the seed. The CPU's and the chips' kernels round differently, so the runs drift apart slowly.

The same check works with `--config=qwen3_1_7b` and the arguments from step 2, plus `--compile.no_enable` for the CPU run, since TorchTPU's `torch.compile` backend isn't loaded there. Its seed checkpoint takes 23 GB with the optimizer state, and the CPU needs about three minutes per step. Over three steps from the same seed:

| Step | 1 | 2 | 3 |
| --- | --- | --- | --- |
| CPU loss | 12.3545 | 10.9783 | 12.2161 |
| Chip loss | 12.3554 | 10.9724 | 12.2709 |
| CPU gradient norm | 4.787 | 6.474 | 41.27 |
| Chip gradient norm | 4.770 | 6.414 | 41.63 |

Both runs show the same jump of the gradient norm in step 3.

## 5. Where the time goes

A step is limited by the chips: the processes queue programs faster than the chips run them, and each chip runs about 17,000 TT-NN programs per step. TT-Metal's per-program device profiler splits a chip's 2.53 seconds like this:

| Work | Time per step | Share |
| --- | --- | --- |
| Attention: score and value matrix multiplies, causal mask, FP32 softmax and their gradients, on [16, 2048, 2048] scores | 0.73 s | 29% |
| Other elementwise work: norms, rotary embeddings, SwiGLU, loss, gradient clipping, AdamW | 0.79 s | 31% |
| Layout changes, slices and dtype casts | 0.48 s | 19% |
| Projection, MLP and output-layer matrix multiplies | 0.26 s | 10% |
| FSDP all-gathers and reduce-scatters | 0.25 s | 10% |
| Embedding gradient (scatter-add into the 151,936-row table) | 0.02 s | 1% |

TorchTitan's `tflops` field, 9.4 per chip, is its FLOPs-per-token estimate (6 per non-embedding parameter, plus attention) times the token rate, without the recomputation of activation checkpointing. All matrix multiplies together, attention's included, take 0.42 seconds, a sixth of the step; most of the time goes to the many smaller elementwise, layout and attention ops around them. A fused attention kernel would remove the largest of those costs: with splash attention unavailable, TorchTPU runs attention as separate matrix multiplies and an FP32 softmax.

libtt's changes on the `torchtitan-training` branch took a step from 6.29 to 2.55 seconds:

| Change | Seconds per step | Tokens per second per chip |
| --- | --- | --- |
| Working baseline | 6.29 | 326 |
| Matrix multiplies read transposed operands in place, instead of permuting attention's score tensors | 4.95 | 414 |
| Permutes drop size-1 dimensions, which tiles padded 32 times | 3.42 | 599 |
| Programs take their arguments tiled, instead of converting them at every program boundary | 3.17 | 646 |
| The same, on C4 with the Qwen3 tokenizer instead of the preset's test data | 3.07 | 666 |
| The fused QKV projection's heads stay as rows of [S × groups, heads × D], instead of tiles padded by the small group dimension | 2.67 | 768 |
| Scatter-adds deal the table's rows out to the cores in turn, so that the embedding gradient's most frequent tokens don't all land on one core | 2.55 | 804 |

The first four rows use the preset's test data and tokenizer, whose token IDs make the embedding gradient slower.

## 6. Known gaps

- libtt's support for TorchTPU on several chips is on a branch and needs a local build until it's released.
- A run that crashes or is killed leaves the chips' ethernet cores down, and the next run stops with `Timed out while waiting for active ethernet core`. Reset the chips with `tt-smi -r` before starting it.
- Splash attention is a Pallas kernel and doesn't run, so attention runs as separate matrix multiplies and an FP32 softmax. It's the largest cost left (see above).
- Each process compiles its programs in the first step of every run, which takes about 20 seconds once TT-Metal's kernels are in its cache (`~/.cache/tt-metal-cache`). libtt doesn't support TorchTPU's persistent compilation cache.
- `torch.manual_seed` doesn't make random draws on the chips repeat, so two runs start from different random weights. Load a seed checkpoint to compare runs exactly.
- TorchTitan's `memory` and `mfu` fields don't describe the Blackhole chips.
- Only the `qwen3_1_7b` and `qwen3_debugmodel` presets with SimpleFSDP (the presets' default for TPUs) are tested. Other models, tensor parallelism and FSDP2 may hit compiler or runtime gaps.

If a run fails, open a [libtt issue](https://github.com/pcmoritz/libtt/issues) with the command, `uv pip freeze`, the libtt commit and the full error of the first process that failed. The [software stack guide](@/docs/software-stack.md#isolate-a-failure) shows how to narrow down a compiler or runtime failure.
