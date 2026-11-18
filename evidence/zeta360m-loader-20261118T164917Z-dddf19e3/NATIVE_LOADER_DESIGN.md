# Native Loader Design and Disposition

**Status: BLOCKED before loader implementation.**

No checkpoint loader was implemented. The pinned Zeta attention component was
executed on synthetic inputs and was found not to preserve the attention
semantics of the pinned Llama checkpoint. Per the task requirement to stop on
an incompatible Zeta component, I did not substitute standard PyTorch/Hugging
Face attention, alter the Zeta repository, map or load model weights, or claim
native inference support.

## Checkpoint Facts

The immutable Hub revision is
`a10cc1512eabd3dde888204e902eca88bddb4951`. Its safetensors header has 290
BF16 tensors and 361,821,120 parameters. Metadata-only inspection matched
every tensor name and shape against the expected 32-layer Llama schema:
288 per-layer tensors (nine projection/norm tensors per layer) plus the
embedding and final norm. There were no missing or unexpected names and no
shape or dtype errors. This recognizes the checkpoint schema; it does **not**
mean the tensors were mapped to or loaded into a native model.

## Blocking Compatibility Fact

At Zeta commit `67d1a34a1ac35d9ea7894ffd9354d5e4e5b7d36b`,
`NonCollapsingAttention` normalizes Q and K and computes attention from
`tau * cosine(Q, K)`. Llama attention for this checkpoint uses scaled raw
dot products (`Q K^T / sqrt(64)`). The real component produced
`0.0000453983` for a synthetic case where the Llama reference produced
`0.33023846` (maximum absolute difference `0.33019307`). This is not a
checkpoint-compatible substitution; a learned scalar tau cannot restore
discarded per-vector magnitudes.

The Zeta RoPE helper and cache can be used on their own under explicit
adaptations/configuration, but cannot make the attention path compatible.
The cache's synthetic sink/recent eviction behavior passed. No operation was
tested with checkpoint activations.

## Required Next Step

The attention blocker needs an explicitly approved Zeta implementation with
the checkpoint's scaled-dot-product semantics, or a separately authorized
change to the Zeta component/API. Only after that compatibility is established
should a loader map the checkpoint tensors and proceed to FP32 resource
qualification. No full model weights were loaded and no model tokens were
processed in this task.
