+++
title = "Software stack"
description = "How PyTorch via torchax and JAX reach Tenstorrent through libtt, and where TorchTPU fits in the roadmap."
weight = 20
+++

## From Python to device execution

The goal is to run existing frameworks and their model implementations out of the box, with little or no application changes. We focus compatibility work on the compiler and runtime integration, reusing upstream framework code instead of maintaining a separate implementation of each model for the hardware.

PyTorch programs run through [torchax](https://github.com/google/torchax), which maps PyTorch operations to JAX. JAX programs use the same backend directly:

```text
PyTorch program → torchax ─┐
JAX program / SGLang-JAX ──┘
  → JAX tracing and StableHLO lowering
  → libtt PJRT plugin (implementation from tt-xla)
  → tt-mlir compilation
  → tt-metal runtime and kernels
  → Tenstorrent device
```

[libtt](https://github.com/pcmoritz/libtt) statically compiles heavily patched upstream libraries [tt-metal](https://github.com/tenstorrent/tt-metal), [tt-mlir](https://github.com/tenstorrent/tt-mlir) and [tt-xla](https://github.com/tenstorrent/tt-xla) into `libtt.so` and a Python wheel. Its local code initializes the plugin, makes the bundled runtime files available, and exposes the PJRT entry points.

## Framework support and TorchTPU

PyTorch via torchax and JAX are supported today. We plan to center PyTorch support on TorchTPU as soon as it is open-sourced.

The backend is experimental, and operation coverage is still in development. The [first experiment](@/docs/first-experiment.md) checks the shared backend with a small JAX program. For PyTorch-specific usage, see the [torchax examples](https://github.com/google/torchax).

## Know the interfaces

| Component | Responsibility | Inspect when |
| --- | --- | --- |
| torchax | Maps PyTorch operations to JAX and provides tensor interoperability. | A PyTorch operation cannot be translated or behaves differently after conversion. |
| JAX | Traces Python array operations and lowers a compiled function to StableHLO. | The traced program has an unexpected shape, dtype, or operation. |
| StableHLO | Represents the tensor computation passed to the compiler. | You need a compiler input independent of the Python application. |
| PJRT / [tt-xla](https://github.com/tenstorrent/tt-xla) | Exposes devices, buffers, compilation, and execution to JAX. | Plugin discovery, device initialization, or buffer handling fails. |
| [tt-mlir](https://github.com/tenstorrent/tt-mlir) | Lowers the input program to Tenstorrent operations and executable artifacts. | An operation is unsupported or compilation produces incorrect code. |
| [tt-metal](https://github.com/tenstorrent/tt-metal) / Metalium | Supplies the runtime, kernels, memory movement, and device execution. | A compiled program hangs, produces incorrect output, or spends time moving data. |

libtt's [reported JAX test suite run](https://github.com/pcmoritz/libtt/blob/b50ce2db8c3dbdebf1ba1818cae833dc472f34e2/README.md) recorded 22,837 passes, 2,800 failures, and 6,531 skips.

## Build and test libtt

For compiler or runtime changes, use a Linux development host with Bazel installed. Start from the [libtt build instructions](https://github.com/pcmoritz/libtt) and record your checkout revision:

```sh
git clone https://github.com/pcmoritz/libtt.git
cd libtt
git rev-parse HEAD
bazel build //:jax_tt_plugin_wheel
bazel test //tests:jax_smoke_tests --test_output=streamed
```

The test target requires a usable Tenstorrent device. Building the wheel includes compiler and runtime dependencies, so allow more time and disk space than for a Python package install.

To collect a JAX test file without opening the device:

```sh
bazel test //tests:jax_test_suite \
  --test_arg=--skip-device-check \
  --test_arg=--collect-only \
  --test_arg=tests/lax_numpy_test.py
```

Collection only lists tests; it does not execute them. Use the repository's full test command when reporting a compatibility baseline so the exclusions are recorded too.

## Isolate a failure

First reproduce the problem with fixed inputs and one compiled function. Export its StableHLO using the [first experiment](@/docs/first-experiment.md), then identify whether failure occurs during lowering, compilation, or execution. Keep the complete error and the last artifact produced successfully.

For wrong results, compare with a host reference before profiling. For slow results, separate compilation, transfers, host dispatch, and device work. For kernel-level investigation, follow the [Metalium and architecture references](../../hardware/#programming-the-device).
