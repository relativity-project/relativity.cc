+++
title = "Docs"
description = "Install libtt, run JAX on a Tenstorrent card, and debug compilation and execution."
sort_by = "weight"
template = "docs/section.html"
page_template = "docs/page.html"
+++

## Choose a starting point

- **New machine:** [Getting started](@/docs/getting-started.md) covers the system driver, Python plugin, and a device check.
- **Working JAX environment:** [First experiment](@/docs/first-experiment.md) checks a matrix multiply against NumPy, times execution, and exports StableHLO.
- **Model serving:** [Inference](@/docs/inference.md) gives a Qwen3-8B launch configuration and an MMLU smoke check.
- **Training:** [Train a tiny Qwen3 model](@/docs/training.md) walks through TorchTitan, TorchAX, SGD updates, and checking a saved checkpoint.
- **Compiler work:** [Software stack](@/docs/software-stack.md) explains the interfaces and what to collect when a program fails.

The hardware examples use a single Blackhole p150a. The training walkthrough uses a tiny model on one device; full-model and multi-card training remain under development.
