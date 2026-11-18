# SmolLM2-360M Architecture Compatibility Investigation

**Disposition: blocked on pinned Zeta attention semantics; no inference was run.**

## Scope and evidence boundary

- Checkpoint: `HuggingFaceTB/SmolLM2-360M-Instruct`, immutable revision
  `a10cc1512eabd3dde888204e902eca88bddb4951`.
- Zeta: `zeta-scaling-toolkit==0.5.1`, pinned source commit
  `67d1a34a1ac35d9ea7894ffd9354d5e4e5b7d36b`.
- The checkpoint `config.json` and safetensors header were inspected. The
  header-only inspection read tensor names, shapes, and dtypes; no tensor
  values were read. It found 290 BF16 tensors and 361,821,120 parameters.
- The source and synthetic tests were inspected/run locally. No model or
  tokenizer was loaded, no checkpoint values were loaded, and no tokens were
  processed.
- Existing 100K and latest blocked qualification evidence was left untouched.
  Their original checksums were verified before adding this report.

The exact pinned config is
`/home/codespace/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951/config.json`.
The checkpoint tensor metadata came from that snapshot's `model.safetensors`
header only. Both report `LlamaForCausalLM`, 32 layers, hidden size 960,
intermediate size 2560, 15 query heads, 5 KV heads, head dimension 64,
`rope_theta=100000`, `rope_interleaved=false`, no RoPE scaling, maximum
positions 8192, vocabulary 49152, SiLU, RMSNorm epsilon `1e-5`, tied word
embeddings, and BF16 storage. See `COMPATIBILITY_MATRIX.json` and
`CHECKPOINT_MAPPING_DESIGN.json` for the detailed comparison.

## Conclusion

**The pinned architecture cannot be supported as genuine Zeta-backed
SmolLM2 inference through a Recall-only adapter.** Recall can implement the
Llama projection, normalization, residual, MLP, embedding, output, and loader
plumbing, and can adapt Zeta's RoPE and cache. It cannot turn the pinned
`NonCollapsingAttention` into the checkpoint's attention without either
changing that Zeta implementation or bypassing it. Bypassing it with
Recall/PyTorch attention would not be genuine execution through pinned Zeta.

The blocker is architectural, not merely a tensor-shape mismatch:

- The checkpoint requires scaled raw dot products,
  `softmax(Q K^T / sqrt(64)) V`.
- Zeta `NonCollapsingAttention.forward` first L2-normalizes Q and K and applies
  a bounded learnable temperature to cosine similarity. Its `d_k=64` input
  and default temperature controls do not reproduce the Llama logits. The
  synthetic compatibility test recorded a maximum absolute output difference
  of `0.3301930726`.
- Its call also rejects unexpanded `[B,15,S,64]` Q and `[B,5,S,64]` K/V because
  the head dimensions are not broadcastable. Repeating each KV head three
  times addresses only this shape constraint, not the attention semantics.

RoPE is a separate, exactly adaptable layout difference: Zeta implements
adjacent interleaved pairs, while the checkpoint is split-half. A fixed
within-head coordinate permutation makes the Zeta RoPE result mathematically
equivalent when applied consistently to Q and K. The synthetic permutation
test had zero maximum absolute error in float64. Zeta's cache is also usable
for one-token decode when instantiated per layer with the checkpoint's
`[B,5,T,64]` KV tensors; for full configured context use `window=8192`,
`sink_tokens=0`, so no key is evicted within the model's 8192-position limit.

The installed Zeta public API exports the attention primitive, RoPE, and cache,
but no checkpoint loader or Llama decoder implementation
(`zeta_scaling_toolkit/__init__.py:1-21`; implementation files are `core.py`
and `positions.py`). A Recall loader/model shell is therefore still needed;
that fact does not resolve the attention incompatibility.

## Required upstream engineering

Before implementing a Recall checkpoint loader or claiming native inference,
the pinned Zeta revision must provide an explicit, tested scaled-dot-product
attention path with grouped-query support. Preserve the existing
`NonCollapsingAttention` behavior for compatibility; preferably add a distinct
primitive or an explicitly selected semantics mode. The new path must accept
15 Q heads and 5 K/V heads, apply the checkpoint's `1/sqrt(64)` scale and
causal visibility, and not normalize away Q/K magnitude or introduce learned
temperature. Requalify that primitive against a direct mathematical
reference before using it.

Then Recall can implement the adapter in `INTEGRATION_DESIGN.md`, map all 290
header-verified tensors, and run future equivalence and resource qualification.
The existing available-memory gate remains **6,144 MiB**. The latest blocked
qualification measured 3,973.46 MiB available and did not load model weights;
this investigation does not change that gate or attempt model loading.

See `BLOCKER_ANALYSIS.md` for the A/B/C assessment and `TEST_RESULTS.json` for
the non-inference checks actually performed.

## Revalidation for this evidence bundle

The checkpoint snapshot was not present at its recorded cache path during this
revalidation. Accordingly, the configuration and tensor-header facts above
are carried forward from the preserved metadata-only inspection in
`evidence/smollm2-360m-architecture-compatibility-20261118T171208Z`; its
SHA-256 manifest was verified. No checkpoint was downloaded or re-read in
this pass. All repository evidence manifests, including the completed 100K
experiment and latest blocked qualification, were also verified without
modifying those directories.

This pass reran the four synthetic Zeta compatibility tests and independently
checked the split-half/interleaved RoPE permutation and synthetic BF16-to-FP32
widening. It did not load a model or checkpoint tensor values and did not run
inference.
