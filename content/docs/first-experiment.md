+++
title = "First experiment"
description = "Turn a successful install into a small, reproducible measurement."
weight = 50
+++

## Pick one observable

Start with a program small enough to understand end to end. Record one observable—correctness, latency, bandwidth, or a compiler artifact—before changing anything.

Good first experiments have:

- fixed input shapes and dtypes
- a deterministic correctness check
- warmup separated from measurement
- exact software revisions
- one variable changed at a time

## Keep the record executable

Store the command, environment, expected result, and observed result together. A useful note should let another person decide in minutes whether their machine behaves the same way.

```text
revision:  <git commit>
hardware:  <card and host>
command:   <exact invocation>
expected:  <correctness or performance bound>
observed:  <result and relevant logs>
```

## Move down only when needed

Stay at the highest layer that can answer the question. Inspect StableHLO when the frontend is suspect, compiler artifacts when lowering is suspect, and Metalium traces when data movement or device execution is suspect.
