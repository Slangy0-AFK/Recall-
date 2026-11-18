# DEEPTRACE_README.md

## Zeta DeepTrace: Native Long-Distance Retrieval Testing Framework

Zeta DeepTrace is a retrieval research and evaluation framework built directly on the **mathematical execution foundation of the Zeta Scaling Toolkit**.

### Purpose

The central question: **Can the existing Zeta mathematical execution architecture retrieve information from increasingly distant positions without relying on an independent retrieval engine?**

DeepTrace measures actual **native retrieval capability** by:

1. Inserting unique facts at known token positions
2. Processing additional intervening information
3. Asking the model to retrieve the original facts
4. Measuring exact-match accuracy across distance bands

This is **not** simply a test of whether Zeta can process 100,000 tokens. It measures whether information introduced earlier can still be **recovered accurately**.

### Key Design Principles

- **Native Retrieval Only**: No external vector databases, RAG systems, or retrieval assistance
- **Deterministic & Reproducible**: All facts, positions, and queries are generated with known random seeds
- **Protected Ground Truth**: Facts are kept from the model during normal execution
- **Cache Diagnostics**: Actual cache status is tracked where technically possible
- **Exact-Match Scoring**: Only perfect answers count as correct
- **Existing Zeta Math**: Reuses actual Zeta components (attention, positional encoding, KV cache)

### Architecture

```
deeptrace/
├── __init__.py                    # Package exports
├── fact_generator.py              # Deterministic fact generation
├── retrieval_dataset.py           # Token stream with facts at known positions
├── retrieval_tests.py             # Five test categories
├── experiment.py                  # Experiment orchestration
└── tests/
    ├── __init__.py
    └── test_deeptrace.py          # Unit tests
```

### Core Components

#### 1. **FactGenerator** (`fact_generator.py`)

Generates deterministic synthetic facts:

- **Equipment codes**: QX-XXXX format (e.g., QX-7319)
- **Location codes**: LM-XXXX format (e.g., LM-4821)
- **Project codes**: PR-XXXX format (e.g., PR-9057)
- **Numeric identifiers**: 5-digit numbers (e.g., 48521)

Each fact has:
- Unique fact ID (ZD-0001, ZD-0002, ...)
- Natural language content
- Known expected answer
- Reproducible with fixed seeds

#### 2. **RetrievalDataset** (`retrieval_dataset.py`)

Builds token streams with facts at known positions:

- Tracks absolute token positions using actual tokenizer
- Records insertion position for each fact
- Generates retrieval questions at specified distances
- Maintains protected ground truth (separate from model access)

#### 3. **Retrieval Tests** (`retrieval_tests.py`)

Five independent test categories:

**Test A: Single-Fact Retrieval**
- Insert one fact, process additional tokens, retrieve its value
- Measure exact-match accuracy

**Test B: Multiple-Fact Retrieval**
- Insert several facts at different positions
- Ask model to retrieve all of them
- Measure individual and complete-answer accuracy

**Test C: Multi-Hop Retrieval**
- Introduce related facts at separate positions
- Ask questions requiring fact connection
- Measure reasoning across distance

**Test D: Aggregation Retrieval**
- Introduce multiple records
- Ask model to combine, count, compare, or summarize
- Measure numerical and categorical accuracy

**Test E: Negative Retrieval**
- Ask about facts never introduced
- Measure false-positive rate and hallucination prevention

#### 4. **DeepTraceExperiment** (`experiment.py`)

Orchestrates complete experiment:

- Generates reproducible test datasets
- Runs positive and negative controls
- Executes approved distance bands
- Scores results with exact-match metric
- Generates comprehensive reports

### Distance Bands

DeepTrace tests at configurable distances:

- **1,024 tokens** (enabled for smoke tests)
- **4,096 tokens** (enabled for smoke tests)
- **8,192 tokens** (requires explicit approval)
- **16,384 tokens** (requires explicit approval)
- **32,768 tokens** (requires explicit approval)
- **65,536 tokens** (requires explicit approval)
- **100,000 tokens** (requires explicit approval)

**Initial configuration runs only 1024 and 4096 token smoke tests.**

Larger distance bands are implemented but disabled until explicitly approved.

### Controls & Validation

**Positive Controls:**
- Short-prompt known-answer tests
- Fact-distractor discrimination
- Repeated-fact consistency

**Negative Controls:**
- Model should not invent answers for unseen facts
- Exact-match should fail on wrong facts
- False information should not be accepted

**Validation Checks:**
- Ground-truth integrity verification
- Token-position verification
- Deterministic seed validation
- Future-information leakage prevention

### Cache & Position Diagnostics

Records:
- Original fact insertion position
- Retrieval question position
- Distance between insertion and question
- Active cache capacity (if observable)
- Cache positions retained
- Whether original fact position has been evicted
- Whether retrieval succeeded or failed
- Model's response (correct, incorrect, or unsupported)

### Output Structure

```
deeptrace/results/
├── config.json                      # Experiment configuration
├── ground_truth.json                # Protected facts (not given to model)
├── smoke_test_results.csv/.json     # Per-query results
├── DEEPTRACE_SUMMARY.json           # Final accuracy report
├── DEEPTRACE_RESULTS.md             # Human-readable summary
└── DEEPTRACE_REPRODUCIBILITY.md     # Commands to reproduce
```

### Scoring

**Overall Exact-Match Accuracy**: Percentage of queries with perfect answer match

**Accuracy by Distance**: Accuracy broken down by token distance band

**Accuracy by Position**: Accuracy broken down by original insertion position

**Control Pass Rate**: Percentage of controls that passed

**False-Positive Rate**: Percentage of negative controls that failed (hallucinated)

### Usage

#### Run Smoke Tests (Demonstration Mode)

```bash
python3 run_smoke_test.py
```

This runs without a model (demonstration-only) to validate the framework.

#### Run with Actual Model

```python
from deeptrace.experiment import DeepTraceConfig, DeepTraceExperiment
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("model_name")
tokenizer = AutoTokenizer.from_pretrained("model_name")

# Configure
config = DeepTraceConfig(
    num_facts=5,
    smoke_test_bands=[1024, 4096],  # Only smoke tests
)

# Run
experiment = DeepTraceExperiment(config=config, model=model, tokenizer=tokenizer)
experiment.run_smoke_tests()
experiment.save_summary()
```

#### Run Unit Tests

```bash
pytest deeptrace/tests/test_deeptrace.py -v
```

### Integration with Zeta Scaling Toolkit

DeepTrace directly uses these Zeta components:

- **NonCollapsingAttention**: QK-normalized attention with bounded temperature
- **StreamingKVCache**: KV cache with sink tokens and sliding window
- **rotary_embedding**: RoPE position encoding
- **alibi_bias**: ALiBi positional biases
- **AsymptoticLRScheduler**: Learning rate scheduler with logarithmic floor

No approximations or simulations are used. All tests execute through Zeta's real mathematical pathway.

### Important Preservation Commitments

This implementation:

✓ Preserves the completed 100K experiment artifacts
✓ Does not restart or resume the completed 100K experiment
✓ Does not overwrite existing benchmark evidence
✓ Does not change published Zeta package behavior
✓ Keeps all DeepTrace outputs in a separate experiment directory
✓ Provides reproducible commands for independent verification

### Expected Outcomes

The framework is designed to answer:

1. **Can Zeta retrieve facts from 1,024 tokens back?** (Smoke test)
2. **Can Zeta retrieve facts from 4,096 tokens back?** (Smoke test)
3. **At what distance does retrieval begin to fail?**
4. **Does cache eviction correlate with retrieval failure?**
5. **Can Zeta avoid hallucinating answers for unseen facts?**
6. **Does accuracy degrade gracefully or catastrophically with distance?**

### Success Criteria

The initial experiment is successful when:

1. Framework correctly identifies facts at known positions
2. Tokenizer position tracking is verified
3. Retrieval questions are generated reproducibly
4. Ground truth remains protected from model access
5. Cache status is measured (where possible)
6. Exact-match scoring works correctly
7. Positive and negative controls pass
8. Smoke tests execute without errors
9. Results are saved independently
10. Another researcher can reproduce the experiment

### Next Steps (Optional)

After smoke tests, optional extensions:

- **Larger distance bands** (8K, 16K, 32K, 65K, 100K)
- **External retrieval comparison** (separate from native test)
- **Cache analysis** (detailed eviction patterns)
- **Model variants** (compare different architectures)
- **Scaling analysis** (effect of model size)

### Limitations

Current version:

- Smoke tests only (1024, 4096 tokens)
- Requires explicit approval for larger distances
- Position tracking accuracy depends on tokenizer
- Cache diagnostics limited to what model exposes
- No automatic recovery from model inference errors

### References

- **Zeta Scaling Toolkit**: https://github.com/Slangy0-AFK/zeta-scaling-toolkit
- **Paper/Methods**: See DEEPTRACE_METHODOLOGY.md
- **Architecture Details**: See DEEPTRACE_ARCHITECTURE.md
- **Reproducibility**: See DEEPTRACE_REPRODUCIBILITY.md

---

**Version**: 0.1.0  
**Status**: Demonstration Mode (Smoke Tests Only)  
**Last Updated**: 2026-11-18
