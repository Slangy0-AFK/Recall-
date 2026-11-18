# DEEPTRACE_REPRODUCIBILITY.md

## Zeta DeepTrace - Reproducibility & Replication Guide

### Overview

This document provides exact instructions for reproducing and replicating the DeepTrace native retrieval tests. It includes commands, dependencies, and expected outputs.

### Environment Setup

#### Prerequisites

- **Python**: 3.10 or later (tested with 3.14.2)
- **PyTorch**: 2.0 or later
- **Git**: For cloning repositories

#### Installation

```bash
# Clone both repositories
git clone https://github.com/Slangy0-AFK/Recall-.git
cd Recall-

# Install dependencies
pip install torch transformers pytest -q

# Optional: Install Zeta Scaling Toolkit for integration
pip install git+https://github.com/Slangy0-AFK/zeta-scaling-toolkit.git
```

### Running Smoke Tests (Framework Validation)

#### Command

```bash
cd /workspaces/Recall-
python3 run_smoke_test.py
```

#### Expected Output

```
======================================================================
ZETA DEEPTRACE - NATIVE RETRIEVAL TEST
======================================================================

[*] Initializing DeepTrace experiment...
[✓] Configuration saved to deeptrace/results/config.json
[*] Running smoke tests (1024 and 4096 tokens)...

======================================================================
DEEPTRACE SMOKE TESTS
======================================================================
[✓] Ground truth saved to deeptrace/results/ground_truth.json

--- Positive Controls ---
  [PASS] short_prompt_known_answer
  [PASS] distractor_discrimination

--- Negative Controls ---
  [PASS] no_hallucination_on_unseen
  [PASS] wrong_facts_fail

--- Test A: Single-Fact Retrieval (Smoke Tests) ---

  Testing distance: 1024 tokens
    [✗] ZD-0001: [answer]
    [✗] ZD-0002: [answer]
    ...

--- Accuracy Report ---
  Overall Accuracy: X.XX%
  Total Queries: 10
  Correct: N
  Accuracy by Distance:
      1024 tokens:  X.XX%
      4096 tokens:  X.XX%

[✓] Results saved to deeptrace/results/smoke_test_results.csv
[✓] Summary saved to deeptrace/results/DEEPTRACE_SUMMARY.json

======================================================================
SMOKE TESTS COMPLETE
======================================================================
```

#### Output Files Generated

```
deeptrace/results/
├── config.json                      # Configuration used
├── ground_truth.json                # Protected ground truth (do not expose to model)
├── smoke_test_results.csv           # Per-query results as CSV
├── smoke_test_results.json          # Per-query results as JSON
└── DEEPTRACE_SUMMARY.json           # Final summary report
```

### Running Unit Tests

#### Command

```bash
cd /workspaces/Recall-
python3 -m pytest deeptrace/tests/test_deeptrace.py -v
```

#### Expected Results

```
======================== 27 passed in X.XXs ========================

Covered areas:
- FactGenerator: generation, determinism, uniqueness
- Distractor generation: generation, determinism
- RetrievalDataset: initialization, fact insertion, position tracking
- RetrievalResult: creation, serialization
- RetrievalScorer: accumulation, accuracy calculation, reporting
```

**All 27 tests must pass** for framework validity.

### Running with an Actual Model

#### Setup (Example with Open-Source Model)

```python
# install_model.py - Install and cache a model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"  # or your chosen model
print(f"Loading {model_name}...")

# Download and cache
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Model loaded and cached.")
```

Run once to cache:
```bash
python3 install_model.py
```

#### Integration Example

```python
# test_with_model.py - Run DeepTrace with actual model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from deeptrace.experiment import DeepTraceConfig, DeepTraceExperiment

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
print(f"Loading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure experiment
config = DeepTraceConfig(
    num_facts=5,
    smoke_test_bands=[1024, 4096],  # Only smoke tests
    output_dir="deeptrace/results",
    seed=42,
)

# Run experiment
print("Initializing experiment...")
experiment = DeepTraceExperiment(config=config, model=model, tokenizer=tokenizer)

print("Running smoke tests...")
experiment.run_smoke_tests()

print("Saving results...")
experiment.save_summary()

print(f"Results in: {config.output_dir}")
```

Run with:
```bash
python3 test_with_model.py
```

### Verifying Reproducibility

#### Same Seed = Same Results

```bash
# Run 1
python3 run_smoke_test.py
cp -r deeptrace/results deeptrace/results_run1

# Run 2 (same seed in config)
python3 run_smoke_test.py
cp -r deeptrace/results deeptrace/results_run2

# Compare
diff deeptrace/results_run1/ground_truth.json deeptrace/results_run2/ground_truth.json
# Should be identical
```

#### Seed Variations

```python
# test_different_seeds.py
from deeptrace.fact_generator import FactGenerator

for seed in [42, 43, 44]:
    gen = FactGenerator(seed=seed)
    facts = gen.generate_dataset(num_facts=3)
    for fact in facts:
        print(f"Seed {seed}: {fact.expected_answer}")
    print()

# Output:
# Seed 42: QX-2824, LM-1409, PR-5506
# Seed 43: QX-6834, LM-8129, PR-2910
# Seed 44: QX-4719, LM-3340, PR-7221
# (Different for each seed)
```

### Checking Output Integrity

#### Ground Truth Validation

```bash
# Verify ground truth exists and is valid JSON
python3 -c "
import json
with open('deeptrace/results/ground_truth.json') as f:
    gt = json.load(f)
    print(f'Facts in ground truth: {len(gt[\"facts\"])}')
    for fact in gt['facts']:
        print(f'  {fact[\"fact_id\"]}: {fact[\"expected_answer\"]}')
"
```

#### CSV Results Validation

```bash
# Verify CSV is well-formed
python3 -c "
import csv
with open('deeptrace/results/smoke_test_results.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    print(f'Total queries: {len(rows)}')
    print(f'Columns: {list(rows[0].keys())}')
    correct = sum(1 for r in rows if r['is_exact_match'] == 'True')
    print(f'Correct: {correct}/{len(rows)}')
"
```

#### JSON Results Validation

```bash
# Verify JSON structure
python3 -c "
import json
with open('deeptrace/results/smoke_test_results.json') as f:
    results = json.load(f)
    print(f'Total results: {len(results)}')
    print(f'First result keys: {list(results[0].keys())}')
    correct = sum(1 for r in results if r['is_exact_match'])
    print(f'Correct: {correct}/{len(results)}')
"
```

### Extending to Larger Distance Bands

**Note**: Only run after explicitly approving extended tests.

#### Configuration

```python
from deeptrace.experiment import DeepTraceConfig, DeepTraceExperiment

config = DeepTraceConfig(
    distance_bands=[1024, 4096, 8192, 16384, 32768, 65536, 100000],
    smoke_test_bands=[1024, 4096],  # Still run smoke tests
    num_facts=5,
    output_dir="deeptrace/results_extended",
)
```

#### Running Extended Tests

```python
# Only enable specific distance bands as approved
experiment = DeepTraceExperiment(config=config, model=model, tokenizer=tokenizer)

# Smoke tests (always safe)
experiment.run_smoke_tests()

# Extended tests (approval required)
# - Implement in experiment.py after approval
# - Add run_extended_tests() method
# - Requires explicit enable flag
```

### Comparison with Baseline

#### Creating a Baseline

```python
# baseline_test.py
# Run one test, save as baseline
from deeptrace.experiment import DeepTraceConfig, DeepTraceExperiment
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("model1")
tokenizer = AutoTokenizer.from_pretrained("model1")

config = DeepTraceConfig(output_dir="deeptrace/results_baseline")
experiment = DeepTraceExperiment(config, model, tokenizer)
experiment.run_smoke_tests()
```

#### Comparing Models

```python
# compare_models.py
import json

models = ["model1", "model2"]
for model_name in models:
    # Load model
    # Run DeepTrace
    # Save results to deeptrace/results_{model_name}/
    pass

# Compare results
for model_name in models:
    with open(f"deeptrace/results_{model_name}/DEEPTRACE_SUMMARY.json") as f:
        summary = json.load(f)
        accuracy = summary["results"]["smoke_tests"]["test_a"]["overall_exact_match_accuracy"]
        print(f"{model_name}: {accuracy:.2%}")
```

### Debugging & Troubleshooting

#### Framework Issues

```bash
# Run unit tests to validate installation
python3 -m pytest deeptrace/tests/test_deeptrace.py -v

# If any test fails:
# 1. Check Python version: python3 --version
# 2. Check dependencies: pip list | grep -E "torch|pytest"
# 3. Check file permissions: ls -la deeptrace/tests/
```

#### Model Issues

```python
# Verify model loads
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    model = AutoModelForCausalLM.from_pretrained("model_name")
    tokenizer = AutoTokenizer.from_pretrained("model_name")
    print("Model loaded successfully")
    
    # Test inference
    input_ids = tokenizer.encode("What is 2+2?", return_tensors="pt")
    output = model.generate(input_ids, max_length=50)
    print("Inference works")
except Exception as e:
    print(f"Error: {e}")
```

#### Position Tracking Issues

```python
# Verify tokenizer position tracking
from deeptrace.retrieval_dataset import RetrievalDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("model_name")
dataset = RetrievalDataset(tokenizer=tokenizer)

# Test text
text = "The assigned equipment code is QX-7319."
tokens = tokenizer.encode(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")
```

### Reporting Results

#### Summary Report Structure

```json
{
  "version": "0.1.0",
  "experiment_type": "DeepTrace Native Retrieval Test",
  "timestamp": "2026-11-18T02:41:00Z",
  "results": {
    "smoke_tests": {
      "test_a": {
        "overall_exact_match_accuracy": 0.XX,
        "accuracy_by_distance": {
          "1024": 0.XX,
          "4096": 0.XX
        },
        "control_pass_rate": 1.0,
        "total_queries": 10,
        "total_correct": N
      }
    }
  },
  "config": {
    "distance_bands": [...],
    "num_facts": 5,
    "seed": 42
  }
}
```

#### Sharing Results

```bash
# Archive complete results
tar -czf deeptrace_results_$(date +%Y%m%d).tar.gz deeptrace/results/

# Share with reviewers
# Include:
# - DEEPTRACE_SUMMARY.json (results)
# - DEEPTRACE_README.md (this file)
# - smoke_test_results.csv (detailed results)
# - config.json (exact configuration)
# DO NOT SHARE:
# - ground_truth.json (keep protected)
```

### Validation Checklist

Before claiming reproducibility, verify:

- [ ] Environment: Python 3.10+, PyTorch 2.0+, Dependencies installed
- [ ] Framework: All 27 unit tests pass
- [ ] Smoke Tests: Run without errors
- [ ] Output Files: All 5 files present and valid JSON/CSV
- [ ] Ground Truth: Protected and not leaked
- [ ] Determinism: Same seed → identical facts
- [ ] Variance: Different seeds → different facts
- [ ] Configuration: Saved and matches execution
- [ ] Commands: All documented commands work
- [ ] Timestamps: Results show recent execution

### Expected Runtime

| Test | Hardware | Time |
|------|----------|------|
| Unit Tests | CPU | ~5 seconds |
| Smoke Tests (no model) | CPU | <1 second |
| Smoke Tests + Model | GPU | 30-60 minutes (depends on model size) |
| Extended Tests | GPU | Hours to days (depends on distance) |

---

**Version**: 0.1.0  
**Status**: Initial Implementation  
**Last Updated**: 2026-11-18
