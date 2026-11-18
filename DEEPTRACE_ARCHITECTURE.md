# DEEPTRACE_ARCHITECTURE.md

## Zeta DeepTrace Architecture

### Overview

Zeta DeepTrace is built as a modular retrieval testing framework that directly integrates with the **Zeta Scaling Toolkit's mathematical execution components**.

```
┌─────────────────────────────────────────────────────────┐
│          DeepTrace Experiment Orchestration              │
│                  (experiment.py)                         │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
  ┌──────────┐  ┌────────────┐  ┌──────────┐
  │   Fact   │  │ Retrieval  │  │   Test   │
  │Generator │  │ Dataset    │  │ Runners  │
  │          │  │ Builder    │  │          │
  └──────────┘  └────────────┘  └──────────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  Zeta Scaling Toolkit       │
        │  (Real Mathematical Path)   │
        └────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
  ┌─────────┐  ┌────────────┐  ┌──────────┐
  │   Non   │  │ Positional │  │    KV    │
  │Collapsing│ │  Encoding  │  │  Cache   │
  │Attention │  │ (RoPE,ALiBi) │          │
  └─────────┘  └────────────┘  └──────────┘
```

### Component Details

#### 1. FactGenerator (`fact_generator.py`)

**Purpose**: Generate deterministic synthetic facts with known answers.

**Key Classes**:
- `Fact`: Data class holding fact_id, content, expected_answer
- `FactGenerator`: Seeds, generates, and saves facts

**Key Methods**:
- `generate_fact(fact_type)`: Create single fact (equipment, location, project, numeric)
- `generate_dataset(num_facts)`: Generate multiple deterministic facts
- `save_ground_truth(facts, path)`: Save protected facts (kept from model)

**Design**:
- Uses `random.Random(seed)` for determinism
- Fact IDs are sequential (ZD-0001, ZD-0002, ...)
- Expected answers are encoded in content (recoverable by parsing)
- Ground truth file is **protected** - should not be given to model during inference

**Example**:
```python
gen = FactGenerator(seed=42)
fact = gen.generate_fact("equipment")
# Returns: Fact(fact_id="ZD-0001", content="The assigned equipment code is QX-7319.",
#               expected_answer="QX-7319")
```

#### 2. RetrievalDataset (`retrieval_dataset.py`)

**Purpose**: Build token streams with facts at known positions.

**Key Classes**:
- `TokenPosition`: Track absolute position, fact_id, is_fact flag
- `RetrievalDataset`: Build and manage token stream

**Key Methods**:
- `add_fact_at_position(fact, context_before)`: Insert fact, record position
- `add_distractors(num_distractors, seed)`: Add unrelated filler text
- `create_retrieval_question(fact, distance_tokens)`: Generate query at known distance
- `get_tokens_as_tensor()`: Return token stream as PyTorch tensor
- `save_manifest(path)`: Save position metadata

**Design**:
- Tracks absolute token position for every token
- Records fact position for retrieval questions
- Supports optional actual tokenizer for precise position tracking
- Falls back to word-level tokenization if tokenizer unavailable
- Separates fact content from surrounding context

**Example**:
```python
dataset = RetrievalDataset(tokenizer=tokenizer)
fact = Fact(..., expected_answer="QX-7319")

# Add fact to stream
fact_pos = dataset.add_fact_at_position(fact, context_before=["Some context"])

# Create retrieval question at distance
query = dataset.create_retrieval_question(fact, distance_tokens=1024)
# Returns: {"fact_id": "ZD-0001", "question": "What is the equipment code?",
#           "fact_insertion_position": 42, "query_position": 1066,
#           "distance_tokens": 1024, "expected_answer": "QX-7319"}
```

#### 3. Retrieval Tests (`retrieval_tests.py`)

**Purpose**: Implement five independent test categories with native Zeta inference.

**Core Class Hierarchy**:
```
RetrievalResult (dataclass)
  ├─ test_id, fact_id
  ├─ original_token_position, query_token_position, distance_tokens
  ├─ expected_answer, actual_answer, is_exact_match
  ├─ cache_status (retained/evicted/unknown)
  ├─ retrieval_latency_ms, control_result
  ├─ zeta_config, git_commit
  └─ to_dict() → serializable dictionary

RetrievalScorer (accumulates results)
  ├─ add_result(result)
  ├─ get_overall_accuracy() → float
  ├─ get_accuracy_by_distance() → Dict[distance, accuracy]
  ├─ get_control_pass_rate() → float
  ├─ generate_report() → Dict
  └─ save_results(path) → CSV + JSON

Test Classes (A through E):
  ├─ TestA_SingleFactRetrieval
  ├─ TestB_MultipleFactRetrieval
  ├─ TestC_MultiHopRetrieval
  ├─ TestD_AggregationRetrieval
  └─ TestE_NegativeRetrieval
```

**Test A: Single-Fact Retrieval**
- Insert 1 fact at position 0
- Process N additional tokens (1024, 4096, ...)
- Ask model: "What is the [identifier type]?"
- Record exact-match result
- Current status: Placeholder (requires actual model inference)

**Test B: Multiple-Fact Retrieval**
- Insert M facts at different positions
- Process additional tokens
- Ask model to retrieve each fact individually
- Measure individual accuracy and complete-answer accuracy
- Current status: Placeholder

**Test C: Multi-Hop Retrieval**
- Insert related facts (e.g., "Project X uses Equipment Y at Location Z")
- Ask connecting questions
- Measure reasoning accuracy
- Current status: Placeholder

**Test D: Aggregation Retrieval**
- Insert multiple records (e.g., multiple equipment codes)
- Ask aggregation question (e.g., "How many unique codes?")
- Measure numerical accuracy
- Current status: Placeholder

**Test E: Negative Retrieval**
- Introduce certain facts
- Ask about facts **never** introduced
- Measure false-positive rate and hallucination prevention
- Current status: Placeholder

#### 4. Experiment Orchestration (`experiment.py`)

**Purpose**: Coordinate complete experiment lifecycle.

**Key Classes**:
- `DeepTraceConfig`: Configuration dataclass (distance bands, fact counts, seeds, etc.)
- `DeepTraceExperiment`: Main orchestrator

**Key Methods**:
- `run_smoke_tests()`: Execute only 1024 and 4096 token tests
- `_run_positive_controls()`: Validate test infrastructure
- `_run_negative_controls()`: Catch hallucinations and false positives
- `save_config()`: Persist configuration
- `generate_summary_report()`: Create final report
- `save_summary()`: Write summary JSON

**Configuration**:
```python
config = DeepTraceConfig(
    distance_bands=[1024, 4096, 8192, 16384, 32768, 65536, 100000],
    smoke_test_bands=[1024, 4096],  # Only these run initially
    num_facts=5,
    fact_types=["equipment", "location", "project", "numeric"],
    num_distractors=10,
    enable_positive_controls=True,
    enable_negative_controls=True,
    cache_window=4096,
    cache_sink_tokens=128,
    seed=42,
    output_dir="deeptrace/results",
)
```

**Experiment Flow**:
1. Initialize experiment with config
2. Generate deterministic facts
3. Save ground truth (protected from model)
4. Build test dataset with facts at known positions
5. Run positive controls to validate infrastructure
6. Run negative controls to catch hallucinations
7. Run Test A (single-fact retrieval) for smoke tests
8. Record results with exact-match scoring
9. Calculate accuracy by distance and position
10. Save results to CSV/JSON/summary
11. Generate final report

### Integration with Zeta Scaling Toolkit

DeepTrace uses these Zeta components:

#### NonCollapsingAttention
- QK-normalized attention with bounded learnable temperature
- Prevents attention collapse on long sequences
- Used in model forward passes
- **Source**: `zeta_scaling_toolkit.NonCollapsingAttention`

#### StreamingKVCache
- Maintains K/V cache with sink tokens and sliding window
- Allows cache diagnostics (what's retained/evicted)
- Supports long sequences without full KV materialization
- **Source**: `zeta_scaling_toolkit.StreamingKVCache`

#### Positional Encodings
- **rotary_embedding**: RoPE with optional scaling (linear/NTK)
- **alibi_bias**: ALiBi positional biases
- Support for extrapolation and length generalization
- **Source**: `zeta_scaling_toolkit.{rotary_embedding, alibi_bias}`

#### Additional Components
- **AsymptoticLRScheduler**: Logarithmic learning rate floor
- **make_corrected_linear**: Weight initialization for depth-aware scaling
- **calibrate_gain**: Gain calibration for layer stability
- **Source**: `zeta_scaling_toolkit.core`

**No Approximations**:
- Tests execute through Zeta's real mathematical pathway
- No mock inference or simulation layers
- Actual tensor operations used throughout
- Cache interactions are real, not simulated

### Data Flow

#### Fact Generation Phase
```
FactGenerator(seed=42)
    ↓
[Fact, Fact, Fact, ...]  (each with fact_id, content, expected_answer)
    ↓
save_ground_truth() → ground_truth.json (protected)
```

#### Dataset Building Phase
```
RetrievalDataset()
    ↓
add_fact_at_position(fact1) → position=0
add_distractors(5) → positions=1..N
add_fact_at_position(fact2) → position=M
...
    ↓
Token stream with position metadata
    ↓
get_tokens_as_tensor() → tensor of shape [sequence_length]
save_manifest() → position_metadata.json
```

#### Query Generation Phase
```
For each fact and distance:
    create_retrieval_question(fact, distance)
        ↓
    Query dict with:
        - fact_id, fact_insertion_position
        - expected_answer
        - question text
        - distance_tokens
```

#### Inference Phase (with model)
```
Token stream + Query → Model inference
    ↓
Model generates response
    ↓
Compare with expected_answer
    ↓
exact_match = (response == expected_answer)
    ↓
RetrievalResult(exact_match=True/False, ...)
```

#### Scoring Phase
```
[Result, Result, Result, ...]
    ↓
RetrievalScorer.add_result(result) × N
    ↓
Scorer accumulates:
    - accuracy_by_distance
    - accuracy_by_position
    - control_pass_rate
    ↓
generate_report() → comprehensive accuracy report
save_results() → CSV + JSON
```

#### Reporting Phase
```
experiment.run_smoke_tests()
    ↓
Test A, Test B, ... (depending on config)
    ↓
Per-test scorer objects
    ↓
experiment.generate_summary_report()
    ↓
Save JSON summary with:
    - version, experiment_type
    - results from all tests
    - notes about disabled distance bands
```

### File Structure

```
deeptrace/
│
├── __init__.py
│   └─ Package initialization and exports
│
├── fact_generator.py
│   ├─ Fact (dataclass)
│   ├─ FactGenerator (class)
│   └─ generate_distractors() (function)
│
├── retrieval_dataset.py
│   ├─ TokenPosition (dataclass)
│   └─ RetrievalDataset (class)
│
├── retrieval_tests.py
│   ├─ RetrievalResult (dataclass)
│   ├─ RetrievalScorer (class)
│   ├─ TestA_SingleFactRetrieval (class)
│   ├─ TestB_MultipleFactRetrieval (class)
│   ├─ TestC_MultiHopRetrieval (class)
│   ├─ TestD_AggregationRetrieval (class)
│   └─ TestE_NegativeRetrieval (class)
│
├── experiment.py
│   ├─ DeepTraceConfig (dataclass)
│   └─ DeepTraceExperiment (class)
│
├── tests/
│   ├─ __init__.py
│   └─ test_deeptrace.py
│       ├─ TestFactGenerator
│       ├─ TestDistractors
│       ├─ TestRetrievalDataset
│       ├─ TestRetrievalResult
│       ├─ TestRetrievalScorer
│       └─ (10+ test methods per class)
│
└── data/  (optional, for caching datasets)
    └─ (generated datasets go here)
```

### Output Files

```
deeptrace/results/
│
├── config.json
│   └─ Full experiment configuration (distance bands, seeds, etc.)
│
├── ground_truth.json
│   ├─ Protected facts with IDs and expected answers
│   └─ NOT given to model during inference
│
├── smoke_test_results.csv
│   └─ Per-query results (test_id, fact_id, distance, expected, actual, match, etc.)
│
├── smoke_test_results.json
│   └─ Same data as CSV but in JSON format
│
├── DEEPTRACE_SUMMARY.json
│   ├─ Experiment version and type
│   ├─ Results from all tests
│   ├─ Accuracy breakdowns
│   └─ Notes about disabled features
│
├── DEEPTRACE_RESULTS.md (future)
│   └─ Human-readable summary with charts and interpretation
│
└── DEEPTRACE_REPRODUCIBILITY.md (future)
    └─ Exact commands to reproduce all tests
```

### Error Handling & Validation

**Validation Checks**:
- Fact IDs must be unique
- Token positions must be non-negative integers
- Expected answers must match content parsing
- Distance must be positive
- Cache capacity must be positive
- Random seeds must be integers

**Fallbacks**:
- If tokenizer unavailable, estimate tokens by word splitting
- If model unavailable, placeholder results (for testing framework)
- If cache diagnostic unavailable, record as "unknown"

**Safety**:
- Ground truth kept completely separate
- Original experiment artifacts never modified
- All results in separate output directory
- Deterministic seeding ensures reproducibility

### Future Extensions

**Planned for Later Approval**:
1. Larger distance bands (8K, 16K, 32K, 65K, 100K)
2. External retrieval comparison (separate experiment)
3. Cache eviction analysis (detailed patterns)
4. Model variants (different architectures)
5. Scaling analysis (effect of model size)

---

**Version**: 0.1.0  
**Status**: Initial Implementation  
**Last Updated**: 2026-11-18
