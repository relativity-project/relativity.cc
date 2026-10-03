+++
title = "Getting started"
description = "Install the system driver and libtt plugin, then run a computation on the TT backend."
weight = 10
+++

## Requirements

You'll need an x86-64 Linux machine with a Blackhole p150a, root access, a network connection, and [uv](https://docs.astral.sh/uv/getting-started/installation/).

Check the [Tenstorrent installation guide](https://docs.tenstorrent.com/getting-started/README.html) for the OS, BIOS, driver and firmware your card needs. Our [p150a dev box](../../hardware/#workstation-configuration) runs Ubuntu 24.04.

## Install the system software

On Ubuntu, install the prerequisites and run Tenstorrent's installer:

```sh
sudo apt update && sudo apt install -y curl jq
/bin/bash -c "$(curl -fsSL https://tenstorrent.ai/install.sh)"
```

The installer sets up the kernel driver, firmware, hugepages and management tools. Reboot when it tells you to. None of this is part of the libtt Python wheel.

If you let the installer create its default Python environment, activate it and look at the card:

```sh
source ~/.tenstorrent-venv/bin/activate
tt-smi
```

`tt-smi` should list every card you installed. (If you picked a different environment, activate that one instead.) Quit `tt-smi` before you continue.

## Install the JAX plugin

Create a separate environment for your experiments:

```sh
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install "jax==0.8.1" "jaxlib==0.8.1" "jax-tt-plugin==0.1.0"
```

These versions match [libtt's reference inference recipe](https://github.com/pcmoritz/libtt/blob/b50ce2db8c3dbdebf1ba1818cae833dc472f34e2/README.md) and the [published plugin wheel](https://pypi.org/project/jax-tt-plugin/0.1.0/). The wheel bundles the compiler and runtime, so you don't need to install tt-metal separately.

## Verify JAX execution

Run this inside the environment:

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

`JAX_PLATFORMS=tt` makes JAX fail loudly if the TT backend can't start, instead of quietly falling back to the CPU. `JAX_USE_SHARDY_PARTITIONER=false` matches libtt's reference configuration. If you see `PASS`, the plugin can open the device, compile a program and return correct results. That's a smoke test; it doesn't mean larger models will work.

## If a check fails

| Symptom | What to try |
| --- | --- |
| `tt-smi` doesn't list the card | Run `lspci -d 1e52:`. If the card is missing there too, it's a hardware or PCIe problem, so fix that before looking at Python packages. If it shows up, check the driver and firmware install. |
| JAX can't initialize `tt` | Make sure the plugin is installed in the active environment: `uv pip show jax-tt-plugin jax jaxlib`. Save the full initialization error. |
| The device opens but compilation fails | Cut the program down to the one failing operation and save its shapes, dtypes and StableHLO. |
| It runs but the values are wrong | Save the reference output and the maximum error, along with the exact dtype and inputs. |

Next, try the [matrix multiply experiment](@/docs/first-experiment.md) or [serve Qwen3-8B](@/docs/inference.md).
