+++
title = "Training"
description = "Train a tiny TorchTitan Qwen3 model on a p150a, save its weights, and check the checkpoint with PyTorch."
weight = 40
+++

## Train a tiny Qwen3 model

This walkthrough trains a **31,987,968-parameter Qwen3 decoder** from scratch on one Blackhole p150a. It uses [TorchTitan's Qwen3 model](https://github.com/pytorch/torchtitan/tree/63de14f5876a195cbe805438614f7b777e1f0daa/torchtitan/models/qwen3), [TorchAX](https://github.com/google/torchax/tree/b2d44f0d6a25db265fe1d93d58dab5de0bf865ca) to run its PyTorch operations through JAX, and libtt to run them on the device. A small TorchAX loop handles the forward pass, gradients and SGD updates.

You'll train on a fixed sequence of token IDs, watch the loss drop, save a checkpoint, and reload it in plain PyTorch to check it. A fixed batch makes learning easy to see, and you don't need to download a tokenizer, dataset or pretrained weights.

Run the [device check](@/docs/getting-started.md#verify-jax-execution) first. You need an x86-64 Linux machine with a working p150a, `uv`, Git and `curl`. Everything runs on one device; TorchTitan's distributed trainer isn't covered here.

## 1. Get the script and dependencies

Download the [training script](../../examples/train_tiny_qwen3.py) and [requirements file](../../examples/training-requirements.txt). Use a fresh environment, since this recipe pins JAX 0.7.1 and specific TorchTitan and TorchAX commits.

```sh
mkdir -p tiny-qwen3
cd tiny-qwen3
curl -fLO https://relativity.cc/examples/train_tiny_qwen3.py
curl -fLO https://relativity.cc/examples/training-requirements.txt
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install "torch==2.13.0+cpu" --index-url https://download.pytorch.org/whl/cpu
uv pip install -r training-requirements.txt
```

The requirements pin the published `jax-tt-plugin==0.1.0` wheel, JAX/jaxlib 0.7.1, Optax 0.2.4, and the TorchTitan and TorchAX commits. The CPU build of PyTorch provides the model API, and TorchAX sends the training computation to the JAX TT backend. Triton is only there because these packages import it.

## 2. Understand the model and task

The script uses TorchTitan's upstream `qwen3_configs["debugmodel"]` preset. TorchTitan defines the architecture and initialization; the script only sets the sequence length and swaps the attention implementation for TorchAX. The settings are:

| Setting | Value |
| --- | --- |
| Transformer layers | 8 |
| Model width / feed-forward width | 256 / 3,072 |
| Attention heads / KV heads | 16 / 8, with 128 elements per head |
| Vocabulary | 2,048 token IDs |
| Batch | One sequence of 32 tokens |
| Parameters | 31,987,968, with shared embedding and output weights |
| Parameter dtype | BF16 |
| Optimizer | SGD, learning rate 0.05 |
| Initialization seed | 0 |

The batch comes from these two lines in the script:

```python
tokens = torch.arange(seq_len, dtype=torch.int64) % vocab_size
labels = (tokens + 1) % vocab_size
```

With the defaults, the inputs are `0, 1, ..., 31` and the targets are `1, 2, ..., 32`. Causal attention lets each position see itself and earlier positions, so repeating this batch teaches the model to predict the next integer. A useful text model would also need varied text, a tokenizer and a held-out evaluation set.

The preset uses FlexAttention, which the script replaces with a small adapter around PyTorch's `scaled_dot_product_attention` so it runs under TorchAX. The adapter transposes TorchTitan's `[tokens, heads, head_dim]` layout to `[heads, tokens, head_dim]`, applies causal grouped-query attention, and transposes back. Everything else (transformer blocks, fused QKV projections, normalization, rotary embeddings, feed-forward layers) comes straight from the upstream preset. The rotary cache is sized for the 32-token sequence.

## 3. Run 20 training steps

In the activated environment:

```sh
env -u TT_METAL_RUNTIME_ROOT \
  python -u train_tiny_qwen3.py \
  --backend tt \
  --steps 20 \
  --output tiny-qwen3.pt
uv pip freeze > training-environment.txt
```

The script selects the TT backend and turns off the Shardy partitioner. The device line should say `TTDevice(id=0, arch=Blackhole)`.

Each step computes the cross-entropy loss, takes gradients for every trainable parameter, and applies an SGD update. This is where the training step is put together:

```python
optimizer = optax.sgd(args.learning_rate)
opt_state = interop.call_jax(optimizer.init, params)
train_step = torchax.train.make_train_step(model_fn, loss_fn, optimizer)
train_step = interop.jax_jit(
    train_step, kwargs_for_jax_jit={"donate_argnums": (0, 2)}
)
```

`JittableModule` gives you a functional version of the TorchTitan model, with parameters and buffers passed in explicitly. The parameters and optimizer state returned by one step feed the next. After moving the model to TorchAX, the script re-ties the shared embedding/output weight so that training and checkpoint loading see the same model.

The loss is computed with an explicit one-hot/log-softmax reduction, which is equivalent to cross-entropy. On the pinned TT backend, `F.cross_entropy` returned wrong loss values in this graph.

Here are the losses from a tested p150a run, with runtime logs and per-step timings left out:

```text
device: TTDevice(id=0, arch=Blackhole)
parameters: 31,987,968
step 01: loss=7.735438
step 05: loss=5.682494
step 10: loss=4.183215
step 20: loss=2.713692
final loss: 2.477034
checkpoint: tiny-qwen3.pt
training: ok
```

Each step's loss is measured before its update; the final loss comes from the saved weights after all 20 updates. The script fails if any loss is non-finite or if the final loss isn't lower than the first. The loss doesn't have to drop at every single step.

The first step includes compilation and can take tens of seconds on an empty cache. Step times include waiting for both the loss and the updated weights on the device. This tiny batch is for learning and debugging; its timings say nothing about full-model training throughput.

## 4. Reload and check the trained weights

The checkpoint is a regular PyTorch state dict plus the model preset, sequence length, seed, step count, learning rate, and the first and final losses. Reload it on the CPU and evaluate the same task there:

```sh
python - <<'PY'
import torch
import torch.nn.functional as F

from train_tiny_qwen3 import build_model, make_batch

checkpoint = torch.load("tiny-qwen3.pt", map_location="cpu", weights_only=True)
config = checkpoint["config"]
model = build_model(**config)
model.load_state_dict(checkpoint["model"])
model.eval()
tokens, labels = make_batch(config["seq_len"], model.config.vocab_size)

with torch.inference_mode():
    logits = model(tokens).float()
    loss = F.cross_entropy(logits, labels)
    correct = (logits.argmax(-1) == labels).sum().item()

torch.testing.assert_close(
    loss, torch.tensor(checkpoint["final_loss"]), rtol=1e-2, atol=1e-2
)
print(f"CPU checkpoint loss: {loss.item():.6f}")
print(f"correct next tokens: {correct}/{labels.numel()}")
print("PASS: checkpoint loss matches TT within tolerance")
PY
```

In the run above, the reloaded checkpoint had a loss of **2.478699** and predicted **28/32** next tokens correctly, so the weights check out in plain PyTorch, not just on the device. Keep in mind this is TorchTitan's debug preset trained on toy data, not a Qwen model you can serve.

## 5. Try another experiment

Once the TT process has exited, run the same training loop on the CPU from the same starting weights:

```sh
python -u train_tiny_qwen3.py \
  --backend cpu \
  --steps 20 \
  --output tiny-qwen3-cpu.pt
```

Compare the loss curves, allowing for BF16 and backend rounding differences. (The checkpoint check above compared the *same trained weights* on two backends; two separate training runs can drift apart as rounding errors add up.)

From here, try more `--steps` or a different `--learning-rate`. The `config` dictionary in `main()` sets the upstream preset, sequence length and seed. On TT, keep the sequence length a multiple of 32. Larger TorchTitan presets need their own memory and execution checks.

To go beyond this fixed batch, add a tokenizer and a train/validation split, feed different batches each step, and evaluate on data the model hasn't seen. Bigger models need memory planning: parameters, gradients, optimizer state, activations and temporary buffers all take space. Adam, for example, adds two moment buffers on top of what SGD needs. Multi-card training needs its own validation of sharding and collectives.

If a run fails, open a [libtt issue](https://github.com/pcmoritz/libtt/issues) with the command, `training-environment.txt`, the model settings, the loss trace and the full error. The [software stack guide](@/docs/software-stack.md#isolate-a-failure) shows how to narrow down a compiler or runtime failure.
