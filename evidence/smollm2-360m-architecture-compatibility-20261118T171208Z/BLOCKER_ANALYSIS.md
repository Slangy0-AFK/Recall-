# Blocker Analysis

## Decision table

| Question | Finding | Reason |
|---|---|---|
| A. Can existing Zeta components support the checkpoint through a Recall adapter? | **No, not as genuine Zeta model execution.** | The Recall adapter can bridge tensor layouts, exact RoPE coordinate layout, and cache API usage, but the pinned `NonCollapsingAttention` changes the attention function from scaled raw dot product to learned-temperature cosine attention. Replacing it with Recall/PyTorch SDPA would bypass Zeta. |
| B. Must Zeta change before native Zeta execution can proceed? | **Yes.** | Zeta needs a distinct exact scaled-dot-product attention path with GQA and causal masking. That capability cannot be added to the pinned binary behavior solely by wrapping the existing API without changing the attention mathematics. |
| C. Does the checkpoint materially differ from the currently supported Zeta architecture? | **Yes, at attention semantics.** | SmolLM2 is a Llama decoder with scaled raw-dot-product GQA, split-half RoPE, RMSNorm and SwiGLU. The current Zeta attention is QK-normalized cosine attention. Tensor shapes alone do not bridge that architectural difference. |

## Core mathematical blocker

For head width 64, checkpoint attention computes
`softmax(Q K^T / 8) V`, where Q and K vector magnitudes affect each score.
Pinned Zeta 0.5.1 `NonCollapsingAttention.forward` first computes
`_unit(q)` and `_unit(k)`, then uses a learnable bounded `tau` over their
cosine similarities. This removes the per-vector magnitudes and changes both
the distribution and its learned temperature. No one fixed `tau` or
permutation can restore all checkpoint scores for arbitrary Q/K inputs.

Synthetic evidence is in `tests/test_zeta_smollm2_compatibility.py` and the
previous compatibility report: the actual Zeta component differed from the
Llama reference by max absolute output error `0.3301930726`. A shape-only
head expansion of five KV heads to 15 query heads passes the Zeta API shape
check but does not change this conclusion.

## Separate, bridgeable differences

- **GQA shape:** exact 3:1 mapping is known; explicit K/V replication may
  satisfy the present Zeta shape contract, but only a new scaled-dot-product
  implementation makes the path semantically correct.
- **RoPE layout:** pinned Zeta's adjacent interleaved layout differs from the
  checkpoint split-half layout. A same-per-head coordinate permutation is an
  exact adapter. A synthetic float64 proof gave zero max absolute error.
- **Positions:** checkpoint uses unscaled integer positions, base 100000, and
  configured max 8192. Zeta accepts int64 positions and that base. Preserve
  absolute positions from cache.
- **Cache:** the cache supports `[B,5,T,64]`, one-token appends and absolute
  positions. Window 8192 with zero sinks retains all keys through the
  configured context, without introducing sink/window attention changes.
- **BF16 to FP32:** widening is exact with respect to the stored BF16 values.
  This does not recover pre-quantization weights and was only tested with
  synthetic values here.
- **Remaining Llama layers and loading:** the pinned Zeta package exposes no
  checkpoint loader, Llama model shell, RMSNorm, SwiGLU, token embedding or
  output layer. Recall can implement this glue and faithfully map the schema,
  but must not represent those operations as Zeta components.

## Next engineering work, in order

1. In Zeta, add and test a separate FP32 scaled raw-dot-product attention API
   that handles 15 Q heads and 5 K/V heads, causal and padding masks, and
   position-aligned cache inputs. Preserve the existing cosine-attention API.
2. Add Zeta unit tests that compare logits/output to a direct SDPA reference
   and explicitly fail if Q/K normalization or learned tau is applied.
3. Implement Recall's metadata-first safetensors mapping design and faithful
   Llama model shell, including RMSNorm, residual order, SwiGLU, tied output
   embedding, exact RoPE permutation, and per-layer cache ownership.
4. Run synthetic primitive and layer equivalence checks without loading the
   checkpoint.
5. Separately re-check the unchanged 6,144 MiB available-memory preflight.
   Only with all code checks passing and that gate satisfied should any
   future checkpoint loading/inference be considered.

No Zeta repository was modified in this investigation. No Recall inference
code or weights were loaded. The latest blocked qualification and the completed
100K experiment remain preserved with their verified original checksums.
