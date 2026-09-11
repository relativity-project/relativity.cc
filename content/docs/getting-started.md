+++
title = "Getting started"
description = "Install the system driver and libtt plugin, then run a computation on the TT backend."
weight = 10
+++

## Requirements

These examples target one Blackhole p150a in an x86-64 Linux host. You need administrator access for system setup, network access for packages, and [uv](https://docs.astral.sh/uv/getting-started/installation/) for the Python environment.

Use the [Tenstorrent installation guide](https://docs.tenstorrent.com/getting-started/README.html) to check OS, BIOS, driver, and firmware requirements for your card. Our [p150a dev box configuration](../../hardware/#workstation-configuration) uses Ubuntu 24.04.

## Install the system software

On Ubuntu, install the prerequisites and start the official installer:

```sh
sudo apt update && sudo apt install -y curl jq
/bin/bash -c "$(curl -fsSL https://tenstorrent.ai/install.sh)"
```

The installer sets up the kernel driver, firmware, hugepages, and management tools. Follow its reboot instructions. These system components are separate from the libtt Python wheel.

If you selected the installer's default Python environment, activate it and inspect the card:

```sh
source ~/.tenstorrent-venv/bin/activate
tt-smi
```

The device count should match the installed cards. If you chose another environment, activate that one instead. Exit `tt-smi` before continuing.

## Install the JAX plugin

Create a separate environment for experiments:

```sh
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install "jax==0.8.1" "jaxlib==0.8.1" "jax-tt-plugin==0.1.0"
```

These pins use the JAX version in [libtt's reference inference recipe](https://github.com/pcmoritz/libtt/blob/b50ce2db8c3dbdebf1ba1818cae833dc472f34e2/README.md) and the [published plugin wheel](https://pypi.org/project/jax-tt-plugin/0.1.0/). The wheel bundles the user-space compiler and runtime; a separate tt-metal installation is not needed.

## Verify JAX execution

Run this from the activated environment:

```sh
JAX_PLATFORMS=tt JAX_USE_SHARDY_PARTITIONER=false python - <<'PY'
import jax
import jax.numpy as jnp
import numpy as np

device = jax.devices("tt")[0]
print("device:", device)
x = jax.device_put(np.arange(32, dtype=np.float32), device)
y = jax.jit(lambda a: a + jnp.float32(1))(x)
y.block_until_ready()
np.testing.assert_array_equal(np.asarray(y), np.arange(1, 33, dtype=np.float32))
print("PASS: addition on", y.device)
PY
```

`JAX_PLATFORMS=tt` requires the TT backend: initialization failure should stop the program instead of silently using the CPU. The second setting follows libtt's reference configuration by disabling the Shardy partitioner. Success means the plugin can initialize a device, compile this operation, and return the expected values. It does not establish coverage for a larger model.

## If a check fails

| Symptom | Check next |
| --- | --- |
| `tt-smi` does not list the card | Run `lspci -d 1e52:`. If PCIe enumeration fails, check hardware setup before Python packages. Otherwise, check the driver and firmware installation. |
| JAX cannot initialize `tt` | Confirm the plugin is installed in the active environment with `uv pip show jax-tt-plugin jax jaxlib`. Save the initialization error. |
| The device opens but compilation fails | Reduce the failing program to one operation and save its shapes, dtypes, and StableHLO. |
| Execution finishes with wrong values | Keep the reference output and maximum error; report the exact dtype and input values. |

Next, run the [matrix multiply experiment](@/docs/first-experiment.md) or follow the [Qwen3-8B recipe](@/docs/inference.md).
