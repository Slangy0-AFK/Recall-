# Runtime Capacity Report

**Status: BLOCKED for inference; PARTIAL for runtime investigation.**

Snapshot: `2026-11-18T16:38:20Z`.

## Current Codespace

The workspace is running in Codespaces on `basicLinux32gb` (2 cores, 8 GB
RAM, 32 GB storage). The guest reports 8,328,056,832 bytes total RAM and
4,325,126,144 bytes available (4,124.762 MiB). There is no swap. The cgroup
memory limit is `max`; its current usage was 3,521,794,048 bytes. The cgroup
CPU quota is unlimited, but the effective CPU set is `0-1`, so this runtime
has two available CPUs. The CPU is an AMD EPYC 7763.

The workspace filesystem is 33,636,024,320 bytes total with 7,271,460,864
bytes free (6.772 GiB).

The existing qualification gate is 6,144 MiB of available RAM. At this
snapshot the gate failed by 2,019.238 MiB. This is a conservative preflight
threshold, not a measured model minimum. No model-load or inference peak RSS
was collected.

## Larger Machine Option

The Codespaces API lists `standardLinux32gb` for this Codespace: 4 cores,
16 GB RAM, and 32 GB storage. It is an available machine-type option, not the
machine currently running. GitHub documents changing the machine type of an
existing Codespace; an active Codespace is stopped and restarted for the
change. The documented CLI form is:

```sh
gh codespace edit --codespace reimagined-yodel-wv7qwjpwjpw72gxx9 --machine standardLinux32gb
```

Changing the machine type can change the billing tier. No resize, purchase,
or other cloud-resource change was made because authorization to incur or
modify that resource was not provided. The current uncommitted worktree was
not moved. A same-Codespace resize is the documented route; verify the
worktree after any authorized change.

## Recommended Safe CPU FP32 Target

- **RAM:** 16 GiB, with at least 6,144 MiB available immediately before
  preflight.
- **CPU:** 4 logical cores minimum for a practical CPU-only run.
- **Storage:** 32 GiB Codespace storage; retain at least 5 GiB free for the
  pinned checkpoint, runtime overhead, and evidence.
- **Precision:** CPU FP32; convert/load only after the memory gate passes,
  avoiding duplicate full-size weight copies.

The listed 4-core/16-GB machine matches this recommendation, but was not
activated. No configuration-file change can increase physical RAM.

## Preservation

The existing blocked run at
`evidence/zeta360m-20261118T162700Z-6effe4ff` was left untouched. The four
pre-existing modified project files remain unedited by this qualification.
No commit or push was performed.

GitHub reference:
[Changing the machine type for your codespace](https://docs.github.com/en/codespaces/customizing-your-codespace/changing-the-machine-type-for-your-codespace).
