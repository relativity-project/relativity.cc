+++
title = "Docs"
description = "Install libtt, run JAX on a Tenstorrent card, and debug what goes wrong."
sort_by = "weight"
template = "docs/section.html"
page_template = "docs/page.html"
+++

## Choose a starting point

- **Setting up a new machine?** [Getting started](@/docs/getting-started.md) installs the driver and the Python plugin, then checks that JAX can use the card.
- **JAX already working?** [First experiment](@/docs/first-experiment.md) runs a matrix multiply, checks it against NumPy, times it, and dumps its StableHLO.
- **Want to serve a model?** [Inference](@/docs/inference.md) has launch commands for one or more chips and a quick MMLU check. The [models page](@/models/_index.md) lists the supported models and how fast they run.
- **Want to train?** [Training](@/docs/training.md) trains a tiny Qwen3 model with TorchTitan and TorchAX, saves a checkpoint, and checks it in PyTorch.
- **Working on the compiler or runtime?** [Software stack](@/docs/software-stack.md) explains how the pieces fit together and what to collect when something breaks.

Most examples use a single Blackhole p150a. Inference also runs across several chips; training currently covers a tiny model on one chip.
