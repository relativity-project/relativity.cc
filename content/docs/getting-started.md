+++
title = "Getting started"
description = "Bring up a development machine and verify that the accelerator is visible."
weight = 10
+++

## Requirements

Start with Ubuntu 24.04 and a supported Tenstorrent card installed in the host. The official installer handles the system packages, drivers, firmware, and development tools needed for the first run.

You will need:

- a supported Tenstorrent accelerator
- Ubuntu 24.04
- administrator access
- an internet connection for packages and repositories

## Install the toolchain

Run the official installer from a clean shell:

```sh
sudo apt update && sudo apt install -y curl jq
/bin/bash -c "$(curl -fsSL https://tenstorrent.ai/install.sh)"
```

Read the prompts before accepting changes. The installer may ask you to log out or reboot when it finishes.

## Verify the device

After the install, confirm that the host can see the card and that the diagnostic checks pass. Keep the output: it is the most useful starting point when debugging a machine that behaves differently from a known-good setup.

Next, read [the software stack](@/docs/software-stack.md) before adding a framework.
