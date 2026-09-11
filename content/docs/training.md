+++
title = "Training"
description = "Experimental work: gradients, optimizer updates, memory use, and full-model validation."
weight = 40
+++

## Current scope

Training is experimental. This site does not yet provide a validated Qwen3-8B training command, convergence result, or training throughput measurement. The [Qwen3-8B example](@/docs/inference.md) covers inference.

A forward pass only exercises part of a training workload. Training also needs backward operations, optimizer state, parameter updates, and enough memory for saved activations. Multi-card training additionally needs working sharding and collectives.

## Validate one training step

Start with the [JAX experiment](@/docs/first-experiment.md) to check device execution and measurement. For a training contribution, use a small model and a fixed batch:

1. Compute the same loss on the CPU and TT backend with identical weights, inputs, and dtypes.
2. Compare each gradient from `jax.value_and_grad`, including its shape and numerical error.
3. Apply one optimizer update and compare the new parameters and optimizer state.
4. Run several steps and record the loss, peak memory use, compilation time, and warm step latency.
5. Repeat at the intended model size and batch size before drawing conclusions about capacity or speed.

Keep inputs fixed while debugging. A decreasing loss alone will not reveal a wrong gradient or an unintended dtype conversion.

## Account for memory

For an 8-billion-parameter model, BF16 parameters alone require about 16 GB in decimal units. BF16 gradients plus two FP32 Adam moment buffers add about 80 GB, bringing those four arrays to roughly 96 GB. That estimate excludes activations, temporary buffers, and any FP32 master weights.

Optimizer choice, sharding, offloading, and activation checkpointing change the requirements. A model fitting for inference does not establish that full-parameter training fits on the same card.

## Useful contributions

| Area | Evidence to include |
| --- | --- |
| Missing backward operation | Minimal `value_and_grad` example, StableHLO, and CPU reference output. |
| Optimizer correctness | Initial state and parameter values before and after one update. |
| Memory reduction | Model and batch size, dtypes, peak memory, and any added recomputation or transfer cost. |
| Distributed training | Device topology, sharding, collective operations, and comparison with a single-device result. |

Attach the exact software revisions and a runnable script to a [libtt issue](https://github.com/pcmoritz/libtt/issues). A full-model recipe should specify the dataset, optimizer, batch size, precision, checkpoints, hardware, and a repeatable loss trace.
