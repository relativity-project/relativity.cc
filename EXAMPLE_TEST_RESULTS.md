# Example validation — 2026-09-11

Tested the runnable examples in `content/docs/` and the local-development commands in `README.md`. Code blocks were extracted from the Markdown and executed from temporary directories. No existing libtt or SGLang-JAX development checkout was modified.

## Environment

- Ubuntu 24.04.4 LTS, x86-64; one Blackhole p150a.
- TT-KMD 2.10.0, firmware 19.13.1, tt-smi 6.1.0.
- Python 3.12.3, uv 0.12.5, Zola 0.23.4, Bazel 9.1.0.
- Getting started / matrix multiply: JAX and jaxlib 0.8.1, published jax-tt-plugin 0.1.0.
- Inference: JAX and jaxlib 0.8.2, published jax-tt-plugin 0.1.0; SGLang-JAX commit `f622d366de3fdaf6965ef4d109b1e6bf861e32a6`.
- Bazel test runner: hermetic Python 3.13.9 and JAX 0.7.1.
- libtt source checkout: `b50ce2db8c3dbdebf1ba1818cae833dc472f34e2`.

## Results

| Example | Result |
| --- | --- |
| System installer | Download and Bash syntax passed; official `--dry-run` completed. No package installation, firmware flashing, or reboot was performed. |
| Card inspection | Activated the documented environment and ran `tt-smi` in noninteractive snapshot mode; one Blackhole card detected. |
| Plugin installation | Passed in a newly created Python environment with the documented version pins. |
| JAX addition | Passed on `TTDevice(id=0, arch=Blackhole)` with exact NumPy equality. |
| Matrix multiplication | Passed the documented `rtol=1e-2, atol=1e-2` check. Maximum absolute error: 0.0153341293. |
| Timing and compiler export | 20 synchronized timed iterations completed; StableHLO file contains `stablehlo.dot_general`; package freeze written. |
| Qwen3-8B server | Started with the documented arguments and loaded the model on the card. |
| Generation request | HTTP 200; 128 completion tokens; response begins with “Paris.” |
| MMLU evaluator | After the dependency fix, all 10 examples completed with four client threads; score 0.8; JSON and HTML reports written. This is a smoke check, not a full accuracy benchmark. |
| Source wheel build | Passed from a clean checkout: `bazel build //:jax_tt_plugin_wheel` (6,808 build actions). |
| libtt smoke tests | Passed: all 17 tests on the p150a using the freshly built wheel. |
| JAX test collection | Passed: 3,290 tests collected. Module imports initialized the TT device despite `--skip-device-check`; corrected the documentation to state that device setup is still required. |
| Site checks | `make check`, `make build`, and `make serve` passed; all eight site pages and the CSS/favicon returned HTTP 200. All 165 internal links/assets across nine generated HTML pages resolved, including fragment identifiers. All 33 external links returned successful HTTP responses. |
| Snippet syntax | All 13 runnable code blocks, including the README, passed Bash/Python syntax checks. |

## Fixes

The MMLU command failed in a clean environment with `ModuleNotFoundError: No module named 'jinja2'`. Adding Jinja2 exposed a second missing dependency, `requests`. Both are now explicit in the command, which also uses `uv run --isolated` to prevent an existing environment from hiding missing dependencies.

The collection instructions incorrectly promised not to open the device. The actual run opened it during test-module imports, so the instructions now explain that `--skip-device-check` does not guarantee device-free collection.

The build instructions now call out the version in libtt's `.bazelversion` and link to Bazelisk. An arbitrary installed Bazel executable is insufficient: this checkout requires 9.1.0.

Raw logs and generated artifacts from this run are in `/tmp/relativity-examples-audit/` on the test host. The temporary inference server was stopped after the checks. Training has no runnable recipe on this site; external repositories linked as further reading were not audited in full.
