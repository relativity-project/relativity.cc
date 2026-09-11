#!/usr/bin/env python3
"""Train a tiny TorchTitan Qwen3 model on libtt through TorchAX."""

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import optax
import torch
import torch.nn.functional as F
import torchax
import torchax.train
from torchax import interop
from torchax.interop import JittableModule
from torchtitan.models.qwen3 import qwen3_configs
from torchtitan.protocols.module import Module


class CausalAttention(Module):
    """Adapt TorchTitan's [tokens, heads, head_dim] layout to PyTorch SDPA."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config):
        super().__init__()

    def forward(
        self, query, key, value, *, attention_masks=None,
        scale=None, enable_gqa=False, **kwargs,
    ):
        if attention_masks is not None:
            raise ValueError("this example uses one sequence with a causal mask")
        output = F.scaled_dot_product_attention(
            query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1),
            is_causal=True, scale=scale, enable_gqa=enable_gqa,
        )
        return output.transpose(0, 1)


def build_model(flavor, seq_len, seed):
    """Use TorchTitan's upstream preset, adapting only attention execution."""
    config = qwen3_configs[flavor](attn_backend="flex")
    for layer in config.layers:
        layer.attention.inner_attention = CausalAttention.Config()
        layer.attention.rope.max_context_length = seq_len
    torch.manual_seed(seed)
    model = config.build().to(dtype=torch.bfloat16)
    model.init_states(buffer_device=torch.device("cpu"))
    return model


def make_batch(seq_len, vocab_size):
    tokens = torch.arange(seq_len, dtype=torch.int64) % vocab_size
    labels = (tokens + 1) % vocab_size
    return tokens, labels


def loss_fn(logits, labels):
    # Equivalent to cross-entropy; avoids incorrect F.cross_entropy loss scalars
    # observed with this graph on the pinned TT backend.
    logits = logits.float()
    targets = F.one_hot(labels, num_classes=logits.shape[-1]).float()
    return -(targets * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def to_cpu(tensor):
    # Copy on the host; NumPy's BF16 dtype needs conversion before torch.from_numpy.
    array = np.array(tensor.jax())
    if tensor.dtype == torch.bfloat16:
        return torch.from_numpy(array.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(array)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="tt", choices=("tt", "cpu"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("tiny-qwen3.pt"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")

    config = {"flavor": "debugmodel", "seq_len": 32, "seed": 0}
    model = build_model(**config).train()
    tokens, labels = make_batch(config["seq_len"], model.config.vocab_size)
    jax.config.update("jax_platforms", args.backend)
    jax.config.update("jax_use_shardy_partitioner", False)
    torchax.enable_performance_mode()
    print(f"device: {jax.devices()[0]}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    with torchax.default_env():
        model = model.to("jax")
        # Moving to TorchAX replaces Parameter objects; restore the shared weight.
        if model.enable_weight_tying:
            model.tok_embeddings.weight = model.lm_head.weight
        jittable_model = JittableModule(model)
        params, buffers = jittable_model.params, jittable_model.buffers
        tokens, labels = tokens.to("jax"), labels.to("jax")

        def model_fn(weights, buffers, tokens):
            return jittable_model.functional_call("forward", weights, buffers, tokens)

        optimizer = optax.sgd(args.learning_rate)
        opt_state = interop.call_jax(optimizer.init, params)
        train_step = torchax.train.make_train_step(model_fn, loss_fn, optimizer)
        train_step = interop.jax_jit(
            train_step, kwargs_for_jax_jit={"donate_argnums": (0, 2)}
        )

        for step in range(1, args.steps + 1):
            start = time.perf_counter()
            loss, params, opt_state = train_step(params, buffers, opt_state, tokens, labels)
            # Include the updated weights, not just the loss, in the step timing.
            jax.block_until_ready(interop.jax_view((loss, params, opt_state)))
            loss = float(loss.item())
            if not math.isfinite(loss):
                raise SystemExit(f"step {step} produced non-finite loss")
            if step == 1:
                initial_loss = loss
            print(f"step {step:02d}: loss={loss:.6f} elapsed={time.perf_counter() - start:.3f}s")

        # Measure the saved weights after the last update.
        @interop.jax_jit
        def evaluate(weights, buffers, tokens, labels):
            return loss_fn(model_fn(weights, buffers, tokens), labels)

        final_loss = float(evaluate(params, buffers, tokens, labels).item())
        if not math.isfinite(final_loss) or final_loss >= initial_loss:
            raise SystemExit(f"loss did not decrease: {initial_loss:.6f} -> {final_loss:.6f}")

        # Install the updated parameters and restore tied aliases for the checkpoint.
        state = jittable_model.functional_call(
            lambda module: module.state_dict(), params, buffers
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {name: to_cpu(value) for name, value in state.items()},
            "config": config,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
        },
        args.output,
    )
    print(f"final loss: {final_loss:.6f}")
    print(f"checkpoint: {args.output}")
    print("training: ok")


if __name__ == "__main__":
    main()
