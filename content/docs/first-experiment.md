+++
title = "First experiment"
description = "Check a matrix multiply against NumPy, measure warm execution, and save its StableHLO."
weight = 25
+++

## Run a matrix multiply

Complete [Getting started](@/docs/getting-started.md), then save this as `matmul.py`. Inputs are generated on the host with a fixed seed and transferred to an explicit TT device.

```python
from pathlib import Path
from statistics import median
from time import perf_counter

import jax
import numpy as np

rng = np.random.default_rng(0)
a_host = rng.uniform(-1, 1, (128, 128)).astype(np.float32)
b_host = rng.uniform(-1, 1, (128, 128)).astype(np.float32)
device = jax.devices("tt")[0]
a = jax.device_put(a_host, device)
b = jax.device_put(b_host, device)
a.block_until_ready()
b.block_until_ready()

matmul = jax.jit(lambda x, y: x @ y)
lowered = matmul.lower(a, b)
Path("matmul.stablehlo.mlir").write_text(
    str(lowered.compiler_ir(dialect="stablehlo"))
)

start = perf_counter()
run = lowered.compile()
compile_s = perf_counter() - start

# Warm up separately; synchronization includes device completion.
for _ in range(3):
    result = run(a, b)
    result.block_until_ready()

actual = np.asarray(result)
expected = a_host @ b_host
max_error = float(np.max(np.abs(actual - expected)))
print("device:", device, "shape:", actual.shape, "dtype:", actual.dtype)
print("maximum absolute error:", max_error)
np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)

samples_ms = []
for _ in range(20):
    start = perf_counter()
    result = run(a, b)
    result.block_until_ready()
    samples_ms.append((perf_counter() - start) * 1000)

print("PASS: matrix multiply within rtol=1e-2, atol=1e-2")
print(f"compile: {compile_s:.3f} s")
print(f"warm execution, median of 20: {median(samples_ms):.3f} ms")
```

Run it in the environment containing the plugin:

```sh
JAX_PLATFORMS=tt JAX_USE_SHARDY_PARTITIONER=false python matmul.py
uv pip freeze > experiment-requirements.txt
```

## Interpret the result

The assertion compares the result with a NumPy computation on the host. Its tolerances are specific to this smoke test, not an accuracy guarantee for other shapes or models. If it fails, keep the measured error and investigate before changing the tolerance.

The timing includes host dispatch and waiting for device completion. It excludes input transfer, NumPy validation, and the explicit compilation call. This small matrix is useful for debugging; it is not a measure of peak accelerator throughput. A compiler cache can also affect the reported compilation time.

JAX dispatch is asynchronous. Omitting `block_until_ready()` can measure dispatch time while the device is still working. See [JAX's benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html).

## Inspect the compiler input

Open `matmul.stablehlo.mlir` and find `stablehlo.dot_general`. Check the operand shapes, element types, and contracting dimensions. Change one input dimension and compare the output. JAX documents this API in [ahead-of-time lowering and compilation](https://docs.jax.dev/en/latest/aot.html).

If lowering succeeds but compilation fails, keep this file with the traceback. If execution returns incorrect values, include the inputs and reference output as well.

## Report a reproducible failure

Include the following in a [libtt issue](https://github.com/pcmoritz/libtt/issues):

- The smallest script and exact invocation that reproduce the result.
- Card model, card count, OS, driver, and firmware versions from your system setup.
- Installed package versions from `experiment-requirements.txt`; the Git commit if you built libtt yourself.
- Expected and observed values, tolerances, and the complete error message.
- For a performance report: shapes, dtypes, warmup count, timed iterations, and whether compilation and transfers are included.

Run the unchanged example before and after a compiler or runtime modification. That gives the change a correctness check and a comparable measurement.
