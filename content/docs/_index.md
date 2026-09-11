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
- **Compiler work:** [Software stack](@/docs/software-stack.md) explains the interfaces and what to collect when a program fails.

The hardware examples use a single Blackhole p150a. Training and multi-card execution are under development; see [training](@/docs/training.md) for the validation work needed before a full-model recipe.
