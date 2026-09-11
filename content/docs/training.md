+++
title = "Training"
description = "Train a tiny TorchTitan Qwen3 model on a p150a, save its weights, and check the checkpoint with PyTorch."
weight = 40
+++

## Train a tiny Qwen3 model

This walkthrough trains a **31,987,968-parameter Qwen3 decoder** from scratch on one Blackhole p150a. It uses [TorchTitan's Qwen3 model](https://github.com/pytorch/torchtitan/tree/63de14f5876a195cbe805438614f7b777e1f0daa/torchtitan/models/qwen3), [TorchAX](https://github.com/google/torchax/tree/b2d44f0d6a25db265fe1d93d58dab5de0bf865ca) to run its PyTorch operations through JAX, and libtt for device execution. A small TorchAX loop handles the forward pass, gradients, and SGD updates.

You will train on a fixed sequence of token IDs, watch the loss fall, save a checkpoint, and reload it in native PyTorch to check the result. The fixed batch makes learning easy to see without downloading a tokenizer, dataset, or pretrained weights.

Complete the [device check](@/docs/getting-started.md#verify-jax-execution) first. You need an x86-64 Linux host, a working p150a, `uv`, Git, and `curl`. This example uses one device; TorchTitan's distributed trainer is outside this walkthrough.

## 1. Get the script and dependencies

Download the [training script](../../examples/train_tiny_qwen3.py) and [requirements file](../../examples/training-requirements.txt). Use a separate environment because this recipe pins JAX 0.7.1 and specific TorchTitan and TorchAX revisions.

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

The requirements include the published `jax-tt-plugin==0.1.0` wheel, JAX/jaxlib 0.7.1, Optax 0.2.4, and the pinned TorchTitan/TorchAX source revisions. PyTorch's CPU build supplies the model API; TorchAX sends the training computation to the JAX TT backend. Triton is included because these upstream packages import it.

## 2. Understand the model and task

The script selects TorchTitan's upstream `qwen3_configs["debugmodel"]` preset. TorchTitan defines the architecture and initialization; the script sets the sequence length and adapts attention for TorchAX. The walkthrough uses:

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

The batch is built with these lines from the script:

```python
tokens = torch.arange(seq_len, dtype=torch.int64) % vocab_size
labels = (tokens + 1) % vocab_size
```

With the defaults, the inputs are `0, 1, ..., 31` and the targets are `1, 2, ..., 32`. Causal attention lets each position see itself and preceding positions. Repeating this batch trains the model to memorize the next integer in the sequence. Training useful text models additionally requires varied text, a tokenizer, and held-out evaluation.

The preset uses FlexAttention. For TorchAX execution, the script replaces that backend with a small adapter to PyTorch's `scaled_dot_product_attention`. It transposes TorchTitan's `[tokens, heads, head_dim]` layout to `[heads, tokens, head_dim]`, applies causal grouped-query attention, and transposes back. The transformer blocks, fused QKV projections, normalization, rotary embeddings, and feed-forward layers come from the upstream preset. The rotary cache is sized to the 32-token training sequence.

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

The script selects the TT backend and disables the Shardy partitioner. Check that the device line says `TTDevice(id=0, arch=Blackhole)`.

Each step computes cross-entropy, differentiates every trainable parameter, and applies an SGD update. These lines assemble the training step:

```python
optimizer = optax.sgd(args.learning_rate)
opt_state = interop.call_jax(optimizer.init, params)
train_step = torchax.train.make_train_step(model_fn, loss_fn, optimizer)
train_step = interop.jax_jit(
    train_step, kwargs_for_jax_jit={"donate_argnums": (0, 2)}
)
```

`JittableModule` supplies a functional version of the TorchTitan model, with parameters and buffers passed explicitly. The returned parameters and optimizer state become the inputs to the next step. The script restores the shared embedding/output weight after moving the model to TorchAX so that training and checkpoint loading preserve the same model.

The loss uses an explicit one-hot/log-softmax reduction equivalent to cross-entropy. This avoids incorrect loss scalars observed with `F.cross_entropy` in this graph on the pinned TT backend.

A tested p150a run produced the following loss values; runtime log lines and per-step timings are omitted here:

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

Each numbered loss is measured before that step's update. The final loss evaluates the saved weights after all 20 updates. The script fails if any loss is non-finite or if the final loss is no lower than the initial loss. Individual steps need not decrease monotonically.

The first step includes compilation and can take tens of seconds with an empty cache. Step timing waits for both the loss and updated weights to finish on the device. This tiny fixed batch is useful for learning and debugging; its timing does not establish full-model training throughput.

## 4. Reload and check the trained weights

The checkpoint contains a regular PyTorch state dictionary, model preset, sequence length, initialization seed, step count, learning rate, and initial/final losses. Reload it on the CPU and evaluate the same task independently:

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

For the run above, the reloaded checkpoint had loss **2.478699** and predicted **28/32** targets correctly. This checks the saved weights with native PyTorch, in addition to the device-side loss check. The checkpoint uses TorchTitan's debug preset and this example's token IDs; it is not a pretrained Qwen model for the inference server.

## 5. Try another experiment

Once the TT process has exited, run the same training loop on the CPU from the same initialization:

```sh
python -u train_tiny_qwen3.py \
  --backend cpu \
  --steps 20 \
  --output tiny-qwen3-cpu.pt
```

Compare the loss curves, allowing for BF16 and backend arithmetic differences. The checkpoint check above compares the *same trained weights* across implementations; two separate training runs can accumulate different rounding errors.

Next, increase `--steps` or adjust `--learning-rate`. The `config` dictionary in `main()` selects the upstream preset, sequence length, and seed. Keep sequence length divisible by 32 on TT. Larger TorchTitan presets need their own memory and execution checks.

When moving beyond this fixed batch, add a tokenizer and train/validation split, feed different batches to the loop, and evaluate on data the model has not seen. Larger models also need memory planning: parameters, gradients, optimizer state, activations, and temporary buffers all contribute. Adam, for example, adds two moment buffers to the SGD setup used here. Multi-card training requires separate validation of sharding and collectives.

If a run fails, keep the command, `training-environment.txt`, model settings, loss trace, and complete error for a [libtt issue](https://github.com/pcmoritz/libtt/issues). The [software stack guide](@/docs/software-stack.md#isolate-a-failure) explains how to reduce a compiler or runtime failure.
