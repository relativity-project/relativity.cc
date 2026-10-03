+++
title = "First experiment"
description = "Check a matrix multiply against NumPy, measure warm execution, and save its StableHLO."
weight = 25
+++

## Run a matrix multiply

Finish [Getting started](@/docs/getting-started.md) first, then save this as `matmul.py`. It generates the inputs on the host with a fixed seed and copies them to a TT device.

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

Run it in the environment where you installed the plugin:

```sh
JAX_PLATFORMS=tt JAX_USE_SHARDY_PARTITIONER=false python matmul.py
uv pip freeze > experiment-requirements.txt
```

## Interpret the result

The assertion checks the result against NumPy on the host. If it fails, save the measured error and find out why before you loosen the tolerance.

The timing covers host dispatch and waiting for the device to finish. It leaves out input transfers, the NumPy check and compilation. A 128×128 matmul is handy for debugging but tells you nothing about peak throughput. A warm compiler cache can also make the compile time look shorter.

JAX dispatches work asynchronously: without `block_until_ready()`, you'd measure how long it takes to queue the work, not to run it. See [JAX's benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html).

## Inspect the compiler input

Open `matmul.stablehlo.mlir` and find `stablehlo.dot_general`. Check the operand shapes, element types and contracting dimensions. Try changing an input dimension and see how the output changes. JAX documents this API under [ahead-of-time lowering and compilation](https://docs.jax.dev/en/latest/aot.html).

If lowering works but compilation fails, keep this file together with the traceback. If execution returns wrong values, also keep the inputs and the reference output.

## Report a reproducible failure

A useful [libtt issue](https://github.com/pcmoritz/libtt/issues) includes:

- The smallest script and the exact command that reproduce the problem.
- Card model and count, OS, driver and firmware versions.
- Package versions, plus the Git commit if you built libtt yourself.
- Expected and actual values, the tolerances, and the full error message.
- For performance problems: shapes, dtypes, the number of warmup and timed iterations, and whether compilation and transfers are included.

Run the unchanged example before and after you modify the compiler or runtime. That gives you a correctness check and a before/after measurement.
