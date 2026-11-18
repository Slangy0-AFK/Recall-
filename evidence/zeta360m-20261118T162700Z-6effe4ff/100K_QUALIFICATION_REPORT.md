# Zeta 360M 100K Qualification

**Status: BLOCKED.** The 100K run did not launch; zero model tokens were
processed.

## Blocking gate

At `2026-11-18T16:29:35Z`, the host reported 3,587,268,608 bytes
(3,421.1 MiB) of available RAM. This is below the existing conservative
6,144 MiB model-load preflight gate by 2,855,182,336 bytes (2,722.9 MiB).
The machine has two logical CPUs, no swap, and 7,272,230,912 bytes of free
disk. The model weights were not loaded, so the actual minimum memory
requirement of a completed native implementation was not measured. The
6,144 MiB value is a preflight gate, not a measured runtime minimum.

The pinned model weights were not present in the checked local Hugging Face
cache. No native 360M inference session was started to establish the pinned
architecture, Zeta component execution, cache behavior, or runtime memory
profile. The architecture and cache gates therefore remain unresolved.

## Qualification disposition

The 32-token smoke stage was blocked by the failed resource gate. The 256,
1,024, 4,096, 8,192, and 16,384-token stages were not run. Checkpoint recovery
and live system integrations were not tested. No model tokens were processed;
the final absolute token position, throughput, wall time, and inference peak
RSS are not applicable.

The 50 tokens/second value is a target, not a measured result. This report
makes no claim of 100K context retention or successful inference.

## Repository safety

The run started on `copilot/zeta-deeptrace-initial-setup` at commit
`6effe4ff3fd7bf4cee44faa2fad521a31d71034a`. The intended branch was absent
locally and on the remote, so `copilot/stack-qualification` was created at
that exact commit. The pre-existing edits in `README.md`,
`config/experiment.json`, `recall_harness.py`, and `tests/test_harness.py`
remain present and unchanged. No historical evidence files were modified.
Nothing was committed or pushed.

See the accompanying JSON records for the exact run configuration, measured
preflight, component trace, cache disposition, checkpoint status, and
integration status. `SHA256SUMS` hashes the files in this new evidence
directory other than itself.
