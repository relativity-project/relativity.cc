+++
title = "Software stack"
description = "How PyTorch via torchax and JAX reach Tenstorrent through libtt, and where TorchTPU fits in the roadmap."
weight = 20
+++

## From Python to device execution

The goal is to run existing frameworks and model code unmodified, or close to it. Rather than porting each model to the hardware, we put the compatibility work into the compiler and runtime and reuse upstream framework code.

PyTorch programs go through [torchax](https://github.com/google/torchax), which maps PyTorch operations onto JAX. JAX programs use the backend directly:

```text
PyTorch program → torchax ─┐
JAX program / SGLang-JAX ──┘
  → JAX tracing and StableHLO lowering
  → libtt PJRT plugin (implementation from tt-xla)
  → tt-mlir compilation
  → tt-metal runtime and kernels
  → Tenstorrent device
```

[libtt](https://github.com/pcmoritz/libtt) statically links heavily patched versions of [tt-metal](https://github.com/tenstorrent/tt-metal), [tt-mlir](https://github.com/tenstorrent/tt-mlir) and [tt-xla](https://github.com/tenstorrent/tt-xla) into a single `libtt.so`, shipped as a Python wheel. Its own code is small: it initializes the plugin, sets up the bundled runtime files, and exposes the PJRT entry points.

## Framework support and TorchTPU

Today you can use JAX directly, or PyTorch through torchax. Once TorchTPU is open-sourced, we plan to build PyTorch support around it.

The backend is experimental and op coverage is still growing. The [first experiment](@/docs/first-experiment.md) exercises the backend with a small JAX program; for PyTorch, start with the [torchax examples](https://github.com/google/torchax).

## Know the interfaces

| Component | What it does | Look here when |
| --- | --- | --- |
| torchax | Maps PyTorch operations to JAX and provides tensor interoperability. | A PyTorch operation cannot be translated or behaves differently after conversion. |
| JAX | Traces Python array operations and lowers a compiled function to StableHLO. | The traced program has an unexpected shape, dtype, or operation. |
| StableHLO | Represents the tensor computation passed to the compiler. | You need a compiler input independent of the Python application. |
| PJRT / [tt-xla](https://github.com/tenstorrent/tt-xla) | Exposes devices, buffers, compilation, and execution to JAX. | Plugin discovery, device initialization, or buffer handling fails. |
| [tt-mlir](https://github.com/tenstorrent/tt-mlir) | Lowers the input program to Tenstorrent operations and executable artifacts. | An operation is unsupported or compilation produces incorrect code. |
| [tt-metal](https://github.com/tenstorrent/tt-metal) / Metalium | Supplies the runtime, kernels, memory movement, and device execution. | A compiled program hangs, produces incorrect output, or spends time moving data. |

In libtt's [last reported run of the JAX test suite](https://github.com/pcmoritz/libtt/blob/b50ce2db8c3dbdebf1ba1818cae833dc472f34e2/README.md), 22,837 tests passed, 2,800 failed and 6,531 were skipped.

## Build and test libtt

To work on the compiler or runtime, you need a Linux machine with the Bazel version pinned in libtt's `.bazelversion` (9.1.0 at the revision tested here); [Bazelisk](https://github.com/bazelbuild/bazelisk) picks it up automatically. Start from the [libtt build instructions](https://github.com/pcmoritz/libtt) and note which commit you're on:

```sh
git clone https://github.com/pcmoritz/libtt.git
cd libtt
git rev-parse HEAD
bazel build //:jax_tt_plugin_wheel
bazel test //tests:jax_smoke_tests --test_output=streamed
```

The test target needs a working Tenstorrent device. The wheel build compiles the whole compiler and runtime, so expect it to take much longer and use far more disk than a normal Python install.

To list the tests in a JAX test file without running them:

```sh
bazel test //tests:jax_test_suite \
  --test_arg=--skip-device-check \
  --test_arg=--collect-only \
  --test_arg=tests/lax_numpy_test.py
```

Collecting tests imports the test modules, which can initialize the TT backend even with `--skip-device-check`, so you still need a device. The tests are listed but not run. When you report a compatibility baseline, use the repository's full test command so the exclusions are recorded too.

## Isolate a failure

Start by reproducing the problem with fixed inputs and a single compiled function. Dump its StableHLO as in the [first experiment](@/docs/first-experiment.md), then work out whether it fails during lowering, compilation or execution. Keep the full error and the last artifact that was produced successfully.

For wrong results, compare against a host reference before you profile anything. For slow results, split the time into compilation, transfers, host dispatch and device work. To dig into kernels, see the [Metalium and architecture references](../../hardware/#programming-the-device).
