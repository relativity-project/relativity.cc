+++
title = "Hardware setup"
description = "A compact host, PCIe dock, and Blackhole p150a development setup."
weight = 40
+++

## Reference machine

The compact workstation described on the hardware page uses:

| Part | Configuration |
| --- | --- |
| Host | Beelink GTi15 Ultra, Intel Core Ultra 9 285H |
| Memory | 64 GB DDR5-5600 |
| Storage | 1 TB SSD |
| Dock | Beelink Multi-Functional EX Pro Docking Station |
| Accelerator | Tenstorrent Blackhole p150a |

The exact host is less important than stable PCIe connectivity, sufficient power, airflow, and a supported Linux installation.

## Before power-on

1. Seat the accelerator and every power connector completely.
2. Confirm that the dock and card have unobstructed airflow.
3. Connect the dock before starting the host.
4. Keep a monitor and keyboard available for the first boot.

## Establish a baseline

Before changing firmware, compiler revisions, or kernel code, save a clean diagnostic run and the versions of every installed component. Repeat the same check after each low-level change.

For the architecture reading path and official references, continue to the [hardware section](@/hardware/_index.md).
