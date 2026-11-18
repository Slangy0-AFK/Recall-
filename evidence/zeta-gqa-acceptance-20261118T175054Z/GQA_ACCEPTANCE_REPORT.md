# Recall-Only Pinned Zeta GQA Acceptance

**Decision: FAIL.** Pinned Zeta 0.5.1 exposes `NonCollapsingAttention`, but it
does not implement SmolLM2-compatible grouped-query attention. Its native
15-query-head/5-KV-head call is rejected. Repeating K/V heads three times
allows a shape-compatible synthetic call, but the result is not numerically
equivalent to standard scaled-dot-product attention.

## Scope and preserved state

- Repository root: `/workspaces/Recall-`
- Branch: `copilot/stack-qualification`
- HEAD at test start: `072f6ebdf7952bc65a9eb6908cde935175eb4959`
- Installed dependency: `zeta-scaling-toolkit==0.5.1`,
  source commit `67d1a34a1ac35d9ea7894ffd9354d5e4e5b7d36b`
- The installed package's `direct_url.json` confirms the pinned commit.
- The five previously preserved evidence manifests were verified, as was the
  additional compatibility bundle from the preceding task. No evidence was
  modified by these checks.
- Only synthetic CPU FP32 tensors were used. No checkpoint, model weights, or
  tokenizer files were loaded; no inference was run. The 6,144 MiB gate was
  not changed. No other repository was modified, and nothing was committed
  or pushed.

## Pinned API inspection

The package exports one attention class:

- Import: `zeta_scaling_toolkit.NonCollapsingAttention`
- Implementation: `zeta_scaling_toolkit/core.py`,
  `NonCollapsingAttention.__init__` and `NonCollapsingAttention.forward`
  (installed source: `/usr/local/python/3.14.2/lib/python3.14/site-packages/zeta_scaling_toolkit/core.py`)
- Constructor:
  `NonCollapsingAttention(d_k, A_attn=1.0, tau0=10.0, tau_max=50.0, tau_min=1.0, check_finite=False, length_scaling='none', train_length=None, tau_eff_max=200.0)`
- Forward:
  `forward(self, q, k, v, attn_mask=None, *, visibility_mask=None)`
- The head dimension is selected by `d_k`; dimension 64 is accepted.
  Attention sequence is the penultimate axis and head axes are leading batch
  axes. There is no exported GQA-specific class/function or `is_causal`
  argument.

`NonCollapsingAttention.forward` unit-normalizes Q and K and applies its
bounded learned `tau` to cosine logits. Its scaled-dot-product kernel path
retains those cosine semantics; it is not raw `QKᵀ / sqrt(64)`. A boolean
`attn_mask` can express visibility, and the synthetic causal-mask test passed
after supplying a lower-triangular mask with `True` meaning visible.

## Mathematical acceptance results

The independent reference implements 3:1 KV grouping, raw QK dot products,
division by `sqrt(64)`, causal masking, softmax, and multiplication by V; it
does not call Zeta attention. Its mapping, finite outputs, causal behavior,
and `[2,4,960]` output shape passed.

The tested Zeta candidate:

1. Rejected Q `[2,15,4,64]` with K/V `[2,5,4,64]` because its leading head
   dimensions are not broadcastable.
2. Accepted an explicit `repeat_interleave(3)` K/V expansion and the causal
   mask. The output was finite, had the expected flattened `[2,4,960]` shape,
   and Q/K/V gradients were finite.
3. Passed causal prefix invariance: changing future K/V did not change earlier
   outputs under the explicit causal mask.
4. **Failed numerical equivalence** against the independent raw scaled
   attention reference at `atol=1e-5`, `rtol=1e-4`: maximum absolute error
   `0.48790717124938965`, mean absolute error `0.04079721122980118`.

Passing shape, causal-mask, or finite-gradient checks does not make the
normalized-cosine operation semantically compatible. Explicit KV expansion
only repairs the head shape; it does not restore the Q/K magnitudes discarded
by normalization.

## Recall integration readiness

`ZetaToolkitAdapter.initialize()` successfully imports the exact pinned
version. Its capability report names only
`zeta_scaling_toolkit.NonCollapsingAttention` and
`StreamingKVCache`; it exposes no GQA operation or model runner. Its component
probe uses one shared `head_count` for Q, K, and V, so it cannot invoke native
15:5 GQA as currently written. Recall therefore **cannot invoke a compatible
Zeta implementation through its existing adapter boundary**.

Genuine SmolLM2 inference is **not technically unblocked**.

## Work required before a 32-token inference smoke test

1. In Zeta, add an explicitly distinct scaled raw-dot-product attention path
   (or semantics mode) that preserves current cosine-attention behavior and
   supports 15 Q heads to 5 K/V heads with 3:1 grouping, `1/sqrt(64)` scaling,
   and causal visibility.
2. In Zeta, test that path against an independent FP32 reference for outputs,
   masks, and gradients; require the acceptance test here to pass.
3. Update Recall's adapter to expose and invoke that API with separate Q-head
   and KV-head dimensions. Keep GQA tensor layout and causal positions
   explicit; do not silently substitute PyTorch attention in Recall.
4. Complete and validate Recall's metadata-first checkpoint mapping, Llama
   residual/RMSNorm/SwiGLU shell, RoPE and cache integration with synthetic
   equivalence tests.
5. Before any authorized smoke test, verify the existing 6,144 MiB available
   memory preflight, then load only the pinned checkpoint/tokenizer and run
   the approved 32-token workload. None of those loads or inference steps were
   performed here.

See `ZETA_API_INSPECTION.json`, `GQA_TEST_RESULTS.json`,
`REFERENCE_COMPARISON.json`, and `RECALL_ADAPTER_READINESS.json` for machine
readable evidence. `run_gqa_acceptance.py` reruns the synthetic checks against
the installed pinned dependency and regenerates the JSON results.
