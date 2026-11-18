# Native SmolLM2-360M Inference Qualification

**Status: BLOCKED before checkpoint loading.**

Run ID: `native-qualification-20261118T170033Z-072f6eb`  
Recall commit: `072f6ebdf7952bc65a9eb6908cde935175eb4959`  
Model: `HuggingFaceTB/SmolLM2-360M-Instruct` at
`a10cc1512eabd3dde888204e902eca88bddb4951`  
Zeta: `zeta-scaling-toolkit==0.5.1` at
`67d1a34a1ac35d9ea7894ffd9354d5e4e5b7d36b`

## Preflight

All eight required model assets were present in the exact pinned snapshot and
their SHA-256 digests matched the pinned digests in the existing provenance
manifest. Metadata-only safetensors inspection found 290 BF16 tensors and
361,821,120 parameters. No tensor values were read.

The existing 6,144 MiB available-RAM safety threshold failed: the preflight
measured 3,973.46 MiB available. The cgroup reported no finite memory limit;
its current usage was 4,688,838,656 bytes. The exact Zeta package revision was
installed, but Recall exposes no checkpoint loader or inference runner. Its
synthetic compatibility tests confirmed that pinned Zeta attention normalizes
Q/K and therefore does not match the checkpoint's scaled raw-dot-product
attention; its RoPE layout also differs from the checkpoint's split-half
layout.

Both blockers were sufficient to stop before model loading. Stage A was not
started, so Stage B was not eligible to run. No model weights or tokenizer were
loaded, and no model tokens were processed. No 100K experiment was launched.

## Execution and evidence boundaries

- **Unit-test execution:** 24 passed, 0 failed, 5 skipped.
- **Legacy pytest execution:** 27 passed, 0 failed, 0 skipped.
- **Synthetic tensor tests:** the four pinned-Zeta compatibility tests passed.
  These establish component behavior and incompatibilities only.
- **Actual checkpoint loading:** not attempted; no model tensor values were
  loaded.
- **Actual native model inference:** not run; 0 model tokens processed.
- **Actual Zeta component execution on model activations:** none. Inference
  attention, RoPE, and cache invocation counts are all 0.

The test command's final qualification status was `BLOCKED` and its process
exit code was 1 because the integration gate is incomplete. The unit and pytest
test runners themselves reported no failures. Five unittest cases were
explicitly skipped; they are not counted as passes.

## Measurements

No inference throughput, inference wall time, inference peak RSS, final model
position, or model cache eviction order exists for this run. Those fields are
null or not run in the accompanying JSON; synthetic cache results are
identified separately and are not represented as model-inference evidence.

Historical evidence checksums passed both before and after the regression
suite. Existing experiment files were not overwritten. No code, configuration,
or historical evidence was modified by this qualification.
