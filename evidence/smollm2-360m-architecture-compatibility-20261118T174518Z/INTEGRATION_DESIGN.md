# Recall / Zeta Integration Design

## Gate

Do not implement checkpoint loading or call this native inference until Zeta
provides a mathematically matching attention primitive. A Recall-only
substitution of PyTorch scaled-dot-product attention would bypass the
incompatible Zeta component and must not be reported as genuine Zeta
execution.

## Proposed Recall module structure

After the upstream attention change, keep the implementation narrow and split
by responsibility:

```text
recall_native/
  checkpoint.py       # immutable revision, metadata-first schema validation, safe loading
  smollm2_360m.py     # Llama model shell, config and token/logit contract
  llama_layer.py      # RMSNorm, residual order, projections, SwiGLU
  zeta_backend.py     # explicit adapters to Zeta attention, RoPE and KV cache
  tests/
    test_checkpoint_schema.py
    test_zeta_attention_equivalence.py
    test_rope_cache_equivalence.py
```

This is a proposal only; no modules were created. Keep the existing Recall
`NativeModel` protocol boundary and make the backend report actual component
identity/revision. Do not vendor or copy Zeta's attention, RoPE, or cache
implementation into Recall.

## Tensor and layer flow

1. Validate immutable checkpoint revision, config fields, and all 290 header
   entries before reading tensor values: two global tensors plus nine tensors
   for each of 32 layers.
2. Load safetensors and widen every BF16 tensor to CPU FP32. Safetensors
   matrices are `[out,in]` and can feed `torch.nn.functional.linear`.
3. Use `model.embed_tokens.weight` with shape `[49152,960]`; apply 32 decoder
   layers at hidden size 960.
4. For each layer, apply input RMSNorm (`eps=1e-5`), Q/K/V projections, RoPE
   on Q/K, exact grouped-query causal attention, output projection, and the
   first residual. Apply post-attention RMSNorm, gate/up projections,
   `down_proj(silu(gate_proj(x))*up_proj(x))`, then the second residual.
5. Apply final RMSNorm. Because `tie_word_embeddings=true` and no
   `lm_head.weight` is stored, compute logits with the same embedding
   parameter as the output projection; do not allocate a separately initialized
   output matrix.

## Zeta attention requirement

Add a distinct and explicit scaled-dot-product path in Zeta while preserving
the existing `NonCollapsingAttention` contract. Its reference operation for
this checkpoint is:

```text
scores = Q @ K.transpose(-2, -1) / sqrt(64)
scores = causal_mask(scores, absolute_query_positions, absolute_key_positions)
output = softmax(scores, dim=-1) @ V
```

Shapes are Q `[B,15,Sq,64]` and K/V `[B,5,Sk,64]`, with each query head
associated with its corresponding one of five KV heads (three query heads per
KV head). Either explicit threefold K/V head expansion or an exact native GQA
kernel is acceptable if numerically equivalent. The new API must not apply Q/K
unit normalization, learned `tau`, or length-dependent temperature. Keep
masking and causal position semantics explicit. The new path must prove
equivalence against a straightforward FP32 reference before Recall uses it.

## RoPE and absolute positions

Use the Zeta `rotary_embedding` primitive with `base=100000`, `scaling=None`,
and `scale_factor=1`. For each 64-wide Q and K head, convert Llama split-half
coordinates to Zeta adjacent-pair coordinates with the permutation
`[0,32,1,33,...,31,63]`; preserve that coordinate system for the attention
dot product. Q and K must use the same permutation. V is not rotated.

Create positions as int64 absolute positions, never by converting a
low-precision counter. The cache's returned absolute positions feed RoPE; with
the checkpoint configuration there is no position scaling and no supported
position beyond 8191.

## KV cache

Create one independent Zeta `StreamingKVCache(window=8192, sink_tokens=0)` per
layer. Append K/V one token at a time in `[B,5,1,64]` form. This retains the
complete key/value history throughout the configured 8192-position context,
so no streaming eviction changes reference attention within that limit. Keep
the cache in FP32, include cache memory in qualification, and test that each
layer's positions stay aligned with the Q RoPE position.

Smaller windows or nonzero sink settings alter vanilla Llama full-context
semantics; do not use them without separately qualifying that intentional
behavioral change.

## Equivalence tests before any model run

- Zeta scaled-dot-product GQA against a direct FP32 PyTorch mathematical
  reference: 15 Q heads / 5 KV heads, 3:1 grouping, multiple sequence lengths,
  causal and padding masks, and invalid shapes. Require numerical tolerances
  set from the backend (initial target `atol=1e-5`, `rtol=1e-4`).
- RoPE permutation against an independently written Llama split-half
  reference for positions 0, 1, 8191 and random positions, full dimension 64,
  base 100000, no scaling.
- Cache path against uncached full-context math for synthetic sequences up to
  a small length and a retention/position test at the 8192 boundary.
- Mapping dry-run verifies every exact checkpoint name/shape/dtype and the
  tied output embedding contract. It must not instantiate weights in the
  dry-run.
- Synthetic layer checks for RMSNorm epsilon, residual ordering, SwiGLU, and
  tied embedding output against direct formulas.
- Only after all above pass: a separate approval-gated model-loading and
  inference qualification, not part of this investigation.

## Resource gate

At 361,821,120 parameters, FP32 parameters alone require 1,447,284,480 bytes
(about 1,380.24 MiB). A batch-one FP32 K/V cache for 32 layers, 5 KV heads,
64 dimensions, and 8192 positions requires 671,088,640 bytes (640 MiB).
These exclude activations, attention temporaries, allocator overhead, the
Python runtime, and input/output buffers; they are not a peak-memory
prediction. Continue to require at least **6,144 MiB available** before any
future model load. Do not lower this threshold to accommodate the current
environment.
