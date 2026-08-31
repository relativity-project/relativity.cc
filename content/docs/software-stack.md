+++
title = "Software stack"
description = "See where JAX, PyTorch/XLA, PJRT, tt-mlir, and tt-metal meet."
weight = 20
+++

## The execution path

An application begins in JAX or PyTorch/XLA and arrives at the device through a deliberately small set of boundaries:

1. **JAX or TorchTPU** produces a program in StableHLO.
2. **libtt** exposes the device through PJRT and accepts the StableHLO program.
3. **tt-mlir** lowers the program toward Tenstorrent operations and device execution.
4. **tt-metal** provides the runtime and kernels that move data and execute work on the card.

This is a map, not a promise that every operation is supported. When something fails, identify the last layer that still represents the program correctly.

## Interfaces worth knowing

### StableHLO

StableHLO is the portable tensor program exchanged at the framework boundary. Inspecting it answers an early question: did the frontend express the computation you intended?

### PJRT

PJRT is the device-facing interface used by JAX and XLA integrations. It covers devices, buffers, compilation, loading, and execution without exposing the compiler implementation to the framework.

### Metalium

Metalium is where device topology, memory movement, circular buffers, and kernels become concrete. This is the layer to study when the program is correct but its execution or performance is not.

## Debug from the boundary inward

Record the StableHLO program, compiler output, runtime logs, and hardware diagnostics separately. A small reproducer at one boundary is more valuable than a large application that crosses all of them.
