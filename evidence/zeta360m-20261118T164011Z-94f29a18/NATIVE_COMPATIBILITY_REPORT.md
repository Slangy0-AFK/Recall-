# Native 360M Compatibility Report

**Status: BLOCKED / UNSUPPORTED in the current Recall implementation.**

## Verified Model Architecture

The pinned Hub revision resolves to the requested commit. Its configuration
identifies `LlamaForCausalLM`: hidden size 960, 32 layers, 15 query heads,
5 key/value heads, 64 dimensions per head, intermediate size 2,560, and
vocabulary size 49,152. RoPE is full-width (64 dimensions), uses
`rope_theta=100000`, has no scaling, and specifies `rope_interleaved=false`.

The safetensors header contains 290 tensors and 361,821,120 parameters.
All 288 expected per-layer projection and norm shapes were checked across
layers 0-31. The Q projection is 960x960; K and V are 320x960 (5 KV heads);
the output projection is 960x960. The gated MLP tensors are 2,560x960,
2,560x960, and 960x2,560. The checkpoint stores BF16 weights; CPU FP32
conversion was not attempted.

## Recall and Pinned Zeta Findings

- `deeptrace/zeta_inference.py` defines a small custom model (defaults:
  hidden size 256, two layers, eight heads). Its K/V projections are
  full-width, its feed-forward layer uses ReLU, and its normalizations are
  LayerNorm. It does not load this Hugging Face checkpoint or map its
  grouped-query attention, gated MLP, RMSNorm, or tied embedding weights.
- `adapters/zeta_adapter.py` describes the model runner and tokenizer as
  unsupported and implements an isolated component probe, not a model loader.
- The pinned Zeta source exports `NonCollapsingAttention`,
  `rotary_embedding`, and `StreamingKVCache`. Source inspection is not
  execution. `zeta_scaling_toolkit` is not installed in the current Python
  runtime, and no Zeta component was called for this qualification.
- Zeta's pinned `rotary_embedding` is interleaved and defaults to base 10,000.
  The model requires non-interleaved RoPE with base 100,000. Although the
  helper accepts a configurable base, its interleaving convention still
  mismatches. No validated adapter translation is present.
- Zeta's cache appends one position at a time and retains sink positions plus
  a recent window. The requested 64 sink + 64 recent configuration can bound
  storage to 128 positions, but the current model-generation code does not
  append to or read from that cache. Its cache-enabled path still forwards the
  full generated sequence.
- `NonCollapsingAttention` consumes Q/K/V with matching per-head dimensions;
  the model's 15 query heads and 5 KV heads need an explicit grouped-query
  mapping. No such mapping or checkpoint weight-loader exists in the current
  Recall model path.

## Conclusion

Architecture metadata and tensor shapes are verified, but successful weight
loading, numerical equivalence, generation, absolute position handling, and
bounded-cache behavior are **not verified**. The existing toy model and
component probe cannot qualify the pinned 360M model. No Hugging Face model
inference fallback was used. Compatibility work and execution must occur in
an authorized adapter after the runtime gate passes.

Inspected source: `deeptrace/zeta_inference.py`,
`adapters/zeta_adapter.py`, and Zeta repository revision
`67d1a34a1ac35d9ea7894ffd9354d5e4e5b7d36b`.
