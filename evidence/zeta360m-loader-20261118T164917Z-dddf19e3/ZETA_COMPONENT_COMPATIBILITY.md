# Pinned Zeta Component Compatibility

**Inspected revision:** `67d1a34a1ac35d9ea7894ffd9354d5e4e5b7d36b`
**Installed test package:** `zeta-scaling-toolkit==0.5.1`, built from that
exact commit. The tests invoked the real package components on small synthetic
tensors; this was not checkpoint inference.

## `NonCollapsingAttention`

- Source: `zeta_scaling_toolkit/core.py`, class
  `NonCollapsingAttention`.
- Constructor includes `d_k`, `tau0=10`, `tau_max=50`, `tau_min=1`,
  `check_finite`, and optional length-scaling controls.
- `forward(q, k, v, attn_mask=None, visibility_mask=None)` requires matching
  final Q/K dimensions and broadcastable leading dimensions.
- The implementation L2-normalizes Q and K, then uses a bounded learnable
  temperature on cosine similarity: logits are `tau * cosine(q, k)`. Its
  optimized path scales normalized Q by `tau * sqrt(d_k)` before
  `scaled_dot_product_attention`, whose internal `1/sqrt(d_k)` scale yields
  the same cosine logits.
- **Not compatible with this checkpoint's attention semantics.** The Llama
  checkpoint expects `softmax(Q K^T / sqrt(64)) V`; Zeta discards Q/K magnitude
  and uses a different, learned temperature. A synthetic real-component run
  measured a `0.33019307` maximum output difference from the Llama formula.
- Unexpanded `[batch, 15, seq, 64]` Q and `[batch, 5, seq, 64]` K/V are not
  broadcastable. Repeating each KV head three times satisfies the API shape,
  but does not fix the different attention mathematics.

## `rotary_embedding`

- Source: `zeta_scaling_toolkit/positions.py`, function
  `rotary_embedding(x, positions, base=10000, scaling=None, scale_factor=1)`.
- `positions` must be `int64`, broadcastable to `x.shape[:-1]`; the last
  dimension must be even. The helper computes frequencies in float64 and
  rotates adjacent interleaved pairs.
- The checkpoint specifies `rope_theta=100000`, full 64-dimensional RoPE,
  and `rope_interleaved=false` (Llama split-half layout). The configurable base
  can be set to 100000, but the pair layout differs. A permutation adapter
  could potentially reconcile the layouts, but none exists or was qualified.
- On synthetic 64-dimensional input with base 100000, direct Zeta RoPE differed
  from a split-half reference by a maximum absolute value of `2.00658466`.

## `StreamingKVCache`

- Source: `zeta_scaling_toolkit/positions.py`, class
  `StreamingKVCache(window, sink_tokens)`.
- `append(key, value)` accepts one position per call (`shape[-2] == 1`),
  detaches stored tensors, returns cached K/V and int64 absolute positions,
  and increments `next_position`.
- With `window=64, sink_tokens=64`, 130 synthetic appends retained positions
  `0..63` and `66..129`, exactly 128 positions, with K/V shapes
  `[1, 5, 128, 64]` and `next_position=130`. The next append used absolute
  position 130.
- Cache insertion, eviction, and finiteness passed with synthetic data. This
  does not establish integration with checkpoint attention or model generation.

## Execution Boundary

Actual Zeta attention, RoPE, and cache methods were invoked by the synthetic
compatibility probe and focused tests. No checkpoint tensor values were
loaded; no model forward, token generation, or checkpoint-backed cache update
occurred. Therefore component execution in this file must not be interpreted
as native model execution.
