# DeepTrace Native Zeta Execution Map

## Objective
Trace the actual execution path of DeepTrace through Zeta Scaling Toolkit's mathematical components during a retrieval test.

---

## 1. Component Discovery and Location

### Zeta Scaling Toolkit Components

| Component | Module | Location | Purpose |
|-----------|--------|----------|---------|
| `NonCollapsingAttention` | `zeta_scaling_toolkit.core` | `/tmp/zeta-scaling-toolkit/zeta_scaling_toolkit/core.py:118-398` | QK-normalized attention with bounded learnable temperature; core LLM attention mechanism |
| `StreamingKVCache` | `zeta_scaling_toolkit.positions` | `/tmp/zeta-scaling-toolkit/zeta_scaling_toolkit/positions.py:159-225` | Detached inference cache with sink tokens and sliding window; manages KV retention across positions |
| `rotary_embedding` | `zeta_scaling_toolkit.positions` | `/tmp/zeta-scaling-toolkit/zeta_scaling_toolkit/positions.py:4-110` | Applies interleaved RoPE position encoding; handles position-aware embeddings |
| `alibi_bias` | `zeta_scaling_toolkit.positions` | `/tmp/zeta-scaling-toolkit/zeta_scaling_toolkit/positions.py:113-156` | Attention linear biases; optional positional bias |
| `streaming_mask` | `zeta_scaling_toolkit.positions` | `/tmp/zeta-scaling-toolkit/zeta_scaling_toolkit/positions.py:227-240` | Mask for cached position visibility |

### Installation
```bash
cd /tmp/zeta-scaling-toolkit && pip install -e .
```

**Status:** ✅ Installed and verified

---

## 2. Execution Path: Input to Output

### Phase 1: Setup
1. **DeepTraceExperiment.__init__()** → Load Zeta components
   - Import `NonCollapsingAttention`, `StreamingKVCache`, `rotary_embedding`
   - Create model with these components
   - Create tokenizer

2. **FactGenerator.generate_dataset()** → Deterministic facts
   - Seed RNG with fixed value
   - Generate 5 facts with known answers

3. **RetrievalDataset()** → Token stream construction
   - Tokenize and position-track each fact
   - Insert facts at known token positions

### Phase 2: Inference for Retrieval Test (Test A)

#### Step 1: Prepare Input
```python
# Input: Token stream with embedded facts
token_ids = tokenizer.encode(
    text_with_facts,  # Original tokens + facts at known positions
    return_tensors="pt"  # Shape: (1, seq_len)
)
positions = torch.arange(seq_len, dtype=torch.int64)  # Absolute token positions
```

**Measurement:** 
- Record input token count
- Record fact positions in token space

#### Step 2: Embedding & Encoding
```python
# Apply rotary position encoding
embeddings = model.embedding(token_ids)  # (batch, seq, d_model)
positions = torch.arange(seq_len, dtype=torch.int64)
embeddings = rotary_embedding(embeddings, positions, base=10000.0)
```

**Measurement:**
- Verify embeddings are finite
- Record dtype (typically float32 or float16)
- Verify position encoding completes without NaN

#### Step 3: Attention Forward Pass
```python
# For each attention layer:
for layer in model.layers:
    # Split heads
    q = embeddings @ W_q  # (batch, seq, d_model) → (batch, num_heads, seq, d_k)
    k = embeddings @ W_k
    v = embeddings @ W_v
    
    # Apply NonCollapsingAttention
    # This is the core Zeta math component
    attn_output = NonCollapsingAttention.forward(
        q, k, v,
        attn_mask=None,  # Can be None for full context
        visibility_mask=None
    )
    # Returns: (batch, num_heads, seq, d_v)
    
    # Combine heads and project
    embeddings = head_combine(attn_output) @ W_o
```

**Zeta Execution Inside `NonCollapsingAttention.forward()`:**

```python
# From zeta_scaling_toolkit/core.py:148-240
def forward(self, q, k, v, attn_mask=None, visibility_mask=None):
    # 1. Normalize q and k (QK-normalization)
    q_unit = _unit(q.to(float32))  # L2-normalize last dim
    k_unit = _unit(k.to(float32))
    
    # 2. Compute unnormalized attention logits
    logits = q_unit @ k_unit.T  # (seq_q, seq_k)
    
    # 3. Scale by learnable temperature
    logits = logits * self.tau  # Bounded temp ∈ [tau_min, tau_max]
    
    # 4. Optional length scaling
    if self.length_scaling == "log":
        counts = attn_mask.sum(dim=-1) if attn_mask else seq_k
        length_factor = log(counts) / log(train_length)
        tau_eff = (self.tau * length_factor).clamp(max=tau_eff_max)
        logits = logits * (tau_eff / self.tau)
    
    # 5. Apply mask (if any facts should be hidden, masked here)
    if attn_mask is not None:
        logits = logits.masked_fill(~attn_mask, -inf)
    
    # 6. Softmax
    attn_weights = softmax(logits, dim=-1)  # (seq_q, seq_k)
    
    # 7. Apply to values
    output = attn_weights @ v  # (seq_q, d_v)
    
    return output.to(original_dtype)
```

**Measurement:**
- Record tau (temperature) values
- Verify logits are finite and bounded
- Record attention weight distribution
- Verify output is finite

#### Step 4: KV Cache Update (if using streaming)
```python
# After generating each token in autoregressive mode:
if use_streaming_cache:
    cache = StreamingKVCache(window=1024, sink_tokens=4)
    
    # Append the new key/value for this position
    new_key = k[:, -1:, :]  # (batch, 1, d_k)
    new_value = v[:, -1:, :]  # (batch, 1, d_v)
    
    cached_k, cached_v, cached_pos = cache.append(new_key, new_value)
    # cached_pos: int64 tensor of which positions are retained
```

**StreamingKVCache.append() Logic (from positions.py:180-212):**

```python
def append(self, key, value):
    position = self.next_position  # Absolute position counter
    
    # Concatenate new key/value with existing cache
    keys = cat([self.keys, key], dim=-2)
    values = cat([self.values, value], dim=-2)
    positions = cat([self.positions, [position]])
    
    # Keep only sink tokens and recent window
    keep = (positions < sink_tokens) | (positions >= next_position - window + 1)
    
    # Evict old positions outside the window
    self.keys = keys[..., keep, :]
    self.values = values[..., keep, :]
    self.positions = positions[keep]
    self.next_position += 1
    
    # Return retained cache state
    return self.keys, self.values, self.positions
```

**Measurement:**
- Record `self.positions` before and after each append
- Identify which fact positions are retained vs. evicted
- Verify cache_positions are int64

#### Step 5: Generation
```python
# Autoregressive token generation
query_text = "What is the answer to our question about " + fact_id + "?"
query_tokens = tokenizer.encode(query_text, return_tensors="pt")

# Append query to context
full_input = cat([token_ids, query_tokens], dim=1)
full_positions = arange(full_input.shape[1], dtype=int64)

# Forward pass with query
with torch.no_grad():
    logits = model(full_input, positions=full_positions)[-1, :, :]
    
# Sample or argmax
answer_token = logits.argmax(dim=-1)
answer = tokenizer.decode(answer_token)
```

**Measurement:**
- Record query token positions
- Verify facts are still cached (or note if evicted)
- Record generated tokens
- Extract expected answer from generation

---

## 3. Cache Eviction Detection

### How to Measure Eviction
```python
# After question generation:

# Record cache state at query time
cached_positions_at_query = cache.positions.tolist()

# Check if fact position is in cache
fact_position = 10  # Example: fact was at token 10
if fact_position in cached_positions_at_query:
    cache_status = "retained"
else:
    cache_status = "evicted"

# Record measurement
result = RetrievalResult(
    ...,
    cache_status=cache_status,
    cache_positions=cached_positions_at_query,
    fact_distance=question_position - fact_position
)
```

### Cache Eviction Timeline (1024-token test)
- Fact inserted at position 0
- Window size: 1024 tokens
- Sink tokens: 4 (keep positions 0-3 always)
- Question at position ~512 (1K test): Fact should be retained
- Question at position ~4096 (4K test): Fact likely evicted (outside window)

---

## 4. Numerical Precision Tracking

### Where Precision Matters
1. **rotary_embedding()**: Position angles computed in float64, then cast to input dtype
2. **NonCollapsingAttention**: Runs in float32 or float64 internally, casts back to input dtype
3. **StreamingKVCache.append()**: Detached tensors (no gradients), positions are int64

### Finiteness Checks
```python
# After each major operation:
for name, tensor in [("embeddings", embeddings), ("logits", logits), ("output", output)]:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or inf")
    
    if tensor.dtype in (torch.float16, torch.float32, torch.float64):
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        # Record for diagnostics
```

---

## 5. Data Flow Diagram

```
Input Stream
    ↓
[Tokenizer] → token_ids, positions (int64)
    ↓
[Embedding] → embeddings (float32/float64)
    ↓
[rotary_embedding()] → position-encoded embeddings
    ↓
[For each Attention Layer]:
    ├─ Split → q, k, v
    ├─ [NonCollapsingAttention.forward()]
    │   ├─ _unit(q) → q_unit (normalized)
    │   ├─ _unit(k) → k_unit (normalized)
    │   ├─ logits = q_unit @ k_unit.T
    │   ├─ logits *= tau (learnable temperature)
    │   ├─ softmax(logits) → attn_weights
    │   └─ attn_weights @ v → output
    ├─ Combine heads
    └─ embeddings ← output
    ↓
[If Streaming Cache]:
    ├─ [StreamingKVCache.append(k, v)]
    │   ├─ Concatenate with previous k, v
    │   ├─ keep = (pos < sink) | (pos >= next_pos - window)
    │   ├─ Evict old positions
    │   └─ Record cached positions
    └─ Continue with cached k, v for future queries
    ↓
[Generate Query/Answer Tokens]
    ├─ Encode question
    ├─ Forward pass with question
    ├─ Sample/argmax → answer
    └─ Record answer
    ↓
[Scoring]
    ├─ Compare answer to ground truth
    ├─ Record exact_match (bool)
    └─ Correlate with cache_status
    ↓
Output Results
```

---

## 6. Test Execution Points

### Test A: 1K Tokens
1. Generate 5 facts (seed=42)
2. Build 1024-token stream with facts at positions [0, 100, 200, 300, 400]
3. For each fact: generate question at distance ~500 tokens
4. Run forward pass through Zeta attention
5. Extract answer
6. Check cache_status and exact_match

### Test B: 4K Tokens
1. Generate 5 facts (seed=43)
2. Build 4096-token stream with facts at known positions
3. For each fact: generate question at distance ~2000 tokens
4. Run forward pass (note: facts likely outside window)
5. Extract answer
6. Record cache eviction evidence

---

## 7. Files Modified vs. Preserved

### Zeta Scaling Toolkit
- **Status:** Read-only, never modified
- **Files touched:** None (imports only)
- **Verification:** `git diff` on /tmp/zeta-scaling-toolkit

### DeepTrace
- **Files modified:**
  - `deeptrace/retrieval_tests.py` → Add real model inference
  - `deeptrace/experiment.py` → Add model loading and execution
- **Files created:** (reports, results)
- **Files preserved:** All existing tests, documentation, 100K experiments

---

## 8. Expected Results

### Pipeline Integrity (PASS/FAIL)
- [ ] Zeta components import without error
- [ ] Model loading completes
- [ ] Token positions are int64 and sequential
- [ ] Embeddings are finite after rotary encoding
- [ ] Attention forward pass completes
- [ ] Cache append completes without shape errors
- [ ] Answer generation completes
- [ ] No numerical anomalies (NaN, inf, underflow)

### Retrieval Capability (1K Test)
- [ ] Fact at position 0 is retrievable at distance ~512
- [ ] Short-distance control passes (should be ~100% accurate)
- [ ] Exact matches recorded correctly
- [ ] Cache status accurately reflects fact position

### Retrieval Capability (4K Test)
- [ ] Fact eviction observed at distance ~2048
- [ ] Long-distance retrieval accurately measured
- [ ] Cache eviction measurement consistent

---

## 9. Instrumentation Checklist

**Before inference:**
- [ ] Model revision recorded
- [ ] Tokenizer identified
- [ ] Seed recorded
- [ ] Fact positions recorded

**During inference:**
- [ ] Input token count
- [ ] tau values per layer
- [ ] Attention logits finite
- [ ] Cache positions retained/evicted
- [ ] Answer tokens generated

**After inference:**
- [ ] Output token count
- [ ] Exact match vs. ground truth
- [ ] Cache status (retained/evicted/unknown)
- [ ] Runtime in milliseconds
- [ ] No exceptions

---

## Summary

This map defines the actual execution path through Zeta's mathematical components:

1. **rotary_embedding()** → Position encodings
2. **NonCollapsingAttention.forward()** → Scaled attention with learnable temperature
3. **StreamingKVCache.append()** → Cache management with eviction

Each component is real, not mocked, and all measurement points are within actual Zeta code execution.
