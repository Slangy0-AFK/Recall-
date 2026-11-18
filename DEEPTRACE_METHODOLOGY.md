# DEEPTRACE_METHODOLOGY.md

## Zeta DeepTrace Testing Methodology

### Research Question

**Can the existing Zeta mathematical execution architecture retrieve information from increasingly distant positions without relying on an independent retrieval engine?**

This is fundamentally different from "Can Zeta process 100,000 tokens?" It measures whether facts **inserted early in a sequence can be accurately recovered** when the model must attend to that information in the presence of intervening tokens.

### Experimental Design

#### Phase 1: Fact Generation (Deterministic)

**Objective**: Create a reproducible set of unique facts with known answers.

**Method**:
1. Use `FactGenerator(seed=42)` for determinism
2. Generate facts in four categories:
   - **Equipment**: Format QX-XXXX (e.g., QX-7319)
   - **Location**: Format LM-XXXX (e.g., LM-4821)
   - **Project**: Format PR-XXXX (e.g., PR-9057)
   - **Numeric**: 5-digit integers (e.g., 48521)

**Example**:
```
FACT ID: ZD-0001
CONTENT: The assigned equipment code is QX-7319.
EXPECTED ANSWER: QX-7319
```

**Key Properties**:
- Each fact has a unique sequential ID
- Expected answer is **deterministically derived** from the random seed
- Answer is embedded in natural language content
- Two different seeds produce completely different facts
- Same seed always produces identical facts (reproducibility)

#### Phase 2: Protected Ground Truth

**Objective**: Maintain a separate, protected dataset of facts that is **never shown to the model during inference**.

**Method**:
1. Save facts to `ground_truth.json` after generation
2. File format:
```json
{
  "facts": [
    {
      "fact_id": "ZD-0001",
      "content": "The assigned equipment code is QX-7319.",
      "expected_answer": "QX-7319"
    },
    ...
  ]
}
```
3. This file is **reference only** - kept completely separate from inference
4. Used after inference to score results

**Preservation**:
- Ground truth is immutable during the experiment
- Answers cannot be leaked to model through context
- Provides definitive scoring source

#### Phase 3: Token Stream Construction

**Objective**: Build a sequence with facts at known positions, using actual tokenizer.

**Method**:
1. Create empty `RetrievalDataset`
2. For each fact:
   - Record absolute token position before insertion
   - Add fact content to stream using real tokenizer
   - Track all token positions
3. Insert distractor information between facts
4. Result: Stream of N tokens with metadata mapping:
   - Token index → Position in document
   - Position in document → Fact ID (if position is part of fact)

**Position Tracking**:
```
Token Index: 0    1    2    3    4    5    6    7    8    9    10   11
Content:     "The" "code" "is" "QX-7319" [distractor text continues...]
Fact:        [------- ZD-0001 insertion position = 0 --------]
```

**Key Details**:
- Uses actual model tokenizer for accurate position calculation
- Never uses character counts as proxy for token counts
- Each token has metadata: `TokenPosition(absolute_pos, fact_id, is_fact)`
- Preserves information about which tokens belong to facts vs. distractors

#### Phase 4: Query Generation

**Objective**: Create retrieval questions at specified distances from facts.

**Method**:
1. For each fact and distance band (1024, 4096, ...):
   - Calculate fact insertion position (from metadata)
   - Calculate query position = insertion_position + distance_tokens
   - Generate natural language question: "What is the [identifier type]?"
   - Create query record:
     ```python
     {
         "query_id": "ZD-0001-query-1024",
         "fact_id": "ZD-0001",
         "question": "What is the equipment code?",
         "fact_insertion_position": 0,
         "query_position": 1024,
         "distance_tokens": 1024,
         "expected_answer": "QX-7319"
     }
     ```

**Deterministic Questions**:
- Question phrasing is consistent for all facts of same type
- Questions are independent of model capabilities
- Expected answers are known in advance
- No adaptive or difficulty-based questioning

#### Phase 5: Controls & Validation

**Positive Controls** (validate infrastructure):

1. **Short-Prompt Known-Answer**
   - Insert fact in first 10 tokens
   - Ask immediately afterward
   - Should have near 100% accuracy if everything works

2. **Distractor Discrimination**
   - Include distractors with similar format (e.g., equipment code-like strings)
   - Verify model selects exact fact, not distractor
   - Prevents false positives from pattern matching

3. **Repeated-Fact Consistency**
   - Insert same fact twice at different positions
   - Both retrievals should match
   - Ensures consistency across distance

**Negative Controls** (catch false positives):

1. **No Hallucination on Unseen**
   - Ask about facts **never** in the stream
   - Model should refuse or admit uncertainty
   - Measures false-positive rate
   - Critical for reliability

2. **Wrong-Fact Rejection**
   - Provide wrong facts as options
   - Model should not accept them
   - Exact-match metric catches this

3. **Future-Information Leakage**
   - Ask about facts that come **after** the query
   - Model should not access future tokens
   - Would indicate data leakage or cheating

**Control Outcomes**:
- If positive controls fail: Results are **inconclusive**
- If negative controls fail: Results are **unreliable** (hallucination)
- Controls must pass for main tests to be meaningful

#### Phase 6: Native Retrieval Tests

**Test A: Single-Fact Retrieval**

Procedure:
1. Insert one fact at known position
2. Add distractors to reach target distance
3. Ask model to retrieve fact value
4. Record: (expected_answer, actual_answer, is_exact_match)

Measurement:
- Exact-match accuracy
- Cache status (retained or evicted)
- Retrieval latency

Success Criteria:
- Accuracy at 1024 tokens: target >80%
- Accuracy at 4096 tokens: target >60%
- Degradation is gradual, not cliff-like

**Test B: Multiple-Fact Retrieval**

Procedure:
1. Insert M facts (5) at different positions
2. Ask model to retrieve each one (or all simultaneously)
3. Record individual accuracies

Measurement:
- Per-fact accuracy
- Complete-answer accuracy (all M correct)
- Interaction effects (does one fact hurt others?)

Success Criteria:
- Individual accuracy remains near Test A levels
- Minimal interaction effects

**Test C: Multi-Hop Retrieval**

Procedure:
1. Insert facts with relationships: "Project X uses Equipment Y at Location Z"
2. Ask connecting questions: "What equipment is used in Project X?"
3. Measure reasoning accuracy

Measurement:
- Accuracy on single-hop reasoning
- Accuracy on multi-hop reasoning
- Chain-of-thought reliability

**Test D: Aggregation Retrieval**

Procedure:
1. Insert multiple records: "Equipment A: QX-100, Equipment B: QX-200, ..."
2. Ask aggregation: "How many equipment codes?", "Compare QX-100 and QX-200..."
3. Measure numerical/categorical accuracy

Measurement:
- Count accuracy
- Comparison accuracy
- Summarization accuracy

**Test E: Negative Retrieval**

Procedure:
1. Insert certain facts
2. Ask about facts **never** introduced
3. Record model response (hallucinated, refused, correct)

Measurement:
- False-positive rate (% of non-facts model claims to know)
- Refusal rate (% model properly declines)
- Hallucination rate (% model invents answers)

Success Criteria:
- Hallucination rate <5%
- Refusal rate >90%

#### Phase 7: Distance Bands

**Testing at Multiple Scales**:

| Distance | Tokens | Purpose | Auto-Run |
|----------|--------|---------|----------|
| 1,024 | 1K | Baseline | ✓ Smoke |
| 4,096 | 4K | Near-window | ✓ Smoke |
| 8,192 | 8K | Medium | ✗ Approval |
| 16,384 | 16K | Long | ✗ Approval |
| 32,768 | 32K | Very Long | ✗ Approval |
| 65,536 | 64K | Extended | ✗ Approval |
| 100,000 | 100K | Maximum | ✗ Approval |

**Smoke Tests** (1K, 4K):
- Run automatically
- Validate framework functionality
- Provide baseline retrieval performance
- Low computational cost

**Extended Tests** (8K+):
- Disabled by default
- Require explicit approval before execution
- Higher computational cost
- May take hours or days

**Distance Analysis**:
1. Plot accuracy vs. distance
2. Identify where accuracy falls below threshold
3. Correlate with cache diagnostics
4. Determine if degradation is gradual or sharp

#### Phase 8: Scoring & Metrics

**Exact-Match Scoring**:
- Only perfect matches count as correct
- No partial credit for similar answers
- Example: "QX-7319" is correct, "QX-7320" is wrong
- Example: "equipment code is QX-7319" is wrong (must extract answer)

**Primary Metrics**:

1. **Overall Exact-Match Accuracy**:
   ```
   Accuracy = (# correct predictions) / (# total queries)
   ```

2. **Accuracy by Distance**:
   ```
   Accuracy(distance=D) = (# correct at distance D) / (# queries at distance D)
   ```

3. **Accuracy by Position**:
   ```
   Accuracy(position=P) = (# correct for facts at position P) / (# queries from position P)
   ```

4. **Control Pass Rate**:
   ```
   Pass Rate = (# passed controls) / (# total controls)
   ```

5. **False-Positive Rate**:
   ```
   FPR = (# hallucinations) / (# negative queries)
   ```

6. **Cache Eviction Correlation**:
   ```
   If observable: Does failure correlate with eviction?
   ```

**Secondary Metrics**:
- Retrieval latency (milliseconds per query)
- Token processing rate (tokens/second)
- Memory usage (MB of cache)
- Consistency (repeated queries)

#### Phase 9: Analysis & Interpretation

**Success Interpretation**:

| Result | Interpretation |
|--------|-----------------|
| High accuracy at 1K, 4K | Retrieval works for near-distance |
| Gradual degradation | Cache-awareness is functioning |
| Accurate at 32K | Long-range retrieval possible |
| Sharp cliff at 32K | Cache eviction boundary |
| Low false-positive rate | Model not hallucinating |
| Passes all controls | Framework is valid |

**Failure Interpretation**:

| Result | Interpretation |
|--------|-----------------|
| Low accuracy at 1K | Framework issue OR model limitation |
| Random accuracy at 4K | Information is not being retained |
| High hallucination rate | Model unreliable for retrieval |
| Failed positive controls | Test infrastructure problem |
| Cache eviction + accuracy drop | Storage limitation, not capability |

**Cache Analysis**:
1. Record which tokens remain in cache
2. Correlate cache retention with retrieval success
3. If token at position P is evicted before query at position Q, can it still be retrieved?
4. Answer indicates whether attention can recover from cache or requires explicit KV retention

### Validation & Quality Assurance

#### Unit Tests (validate components)
- `test_deeptrace.py`: 20+ unit tests covering:
  - Fact generation determinism
  - Distractor generation
  - Token position tracking
  - Query generation
  - Result scoring
  - Report generation

#### Integration Tests (validate workflow)
- End-to-end smoke test without model
- Verify dataset construction
- Verify file outputs
- Verify manifest consistency

#### Reproducibility Verification
- Run identical experiment twice with same seed
- Results must be identical
- Proves determinism

#### Ground Truth Integrity
- Verify ground truth file exists before inference
- Verify no ground truth data leaked to model
- Verify scoring uses protected ground truth
- Verify results file doesn't expose answers

### Limitations & Caveats

**Framework Limitations**:
1. Requires actual model for real tests (not just unit tests)
2. Position tracking accuracy depends on tokenizer
3. Cache diagnostics limited to what model exposes
4. No automatic error recovery

**Experimental Limitations**:
1. Synthetic facts may not match real language distribution
2. Distractor text is controlled, not naturally distributed
3. Questions are deterministic, not adaptive
4. Model may have training data contamination (esp. for codes like QX-XXXX)

**Interpretive Limitations**:
1. Failure at 4K doesn't mean model can't do retrieval (could be training issue)
2. Success at 100K doesn't prove general retrieval (could be pattern matching)
3. Cache eviction may not be sole cause of failure
4. Single test run may have high variance

**Mitigations**:
- Multiple runs with different seeds
- Comparison with baseline models
- Analysis of failure modes
- Validation against other retrieval tests

### Success & Completion Criteria

The experiment is successful when:

1. ✓ Framework correctly generates reproducible facts
2. ✓ Dataset construction tracks token positions accurately
3. ✓ Retrieval questions are generated at known distances
4. ✓ Ground truth remains protected from model
5. ✓ Cache status is measured (where possible)
6. ✓ Exact-match scoring is implemented
7. ✓ Positive controls pass (validate infrastructure)
8. ✓ Negative controls pass (catch hallucinations)
9. ✓ Smoke tests execute without errors
10. ✓ Results are saved and reproducible
11. ✓ All files and code are documented
12. ✓ Another researcher can reproduce the experiment

The experiment is inconclusive if:
- Positive controls fail
- Multiple runs show high variance
- Framework issues are discovered

The experiment is invalid if:
- Ground truth leaks to model
- Tokenizer position tracking is inaccurate
- Random seed doesn't produce determinism

---

**Version**: 0.1.0  
**Status**: Initial Implementation  
**Last Updated**: 2026-11-18
