# Recall-

Zeta DeepTrace is a retrieval research and evaluation framework intended to measure
long-distance information retrieval using the existing mathematical execution
foundation of the Zeta Scaling Toolkit.

## Current scope

This repository does not contain the Zeta model runtime, tokenizer, or KV-cache
implementation. DeepTrace therefore exposes a small adapter contract rather
than reimplementing inference. The current harness supports deterministic
synthetic needle and absent-fact controls, tokenizer-counted positions,
per-question native-model evaluation, cache-position observations when provided
by the adapter, and JSON, JSONL, CSV, and Markdown artifacts.

The harness labels results `native_zeta` and never adds external retrieval.
It does not infer cache eviction when the adapter omits position metadata.
Multi-hop, aggregation, controlled-retention, and external-retrieval comparison
experiments are not implemented yet; they must remain separately identified
when added.

## Backend adapter

Implement `Tokenizer.encode(text)` using the tokenizer used by the model, and
adapt the existing model execution path to the following synchronous interface:

```python
class ZetaAdapter:
    def reset(self):
        # Reset the model's normal per-example execution state.
        ...

    def prefill(self, text):
        # Process text through the existing Zeta execution path.
        ...

    def generate(self, prompt):
        # Continue from that model state and return the generated answer.
        ...

    def cache_metadata(self):
        # Optionally return positions, or start_position/end_position,
        # from the real active cache; do not modify cache behavior here.
        return {}

    def metadata(self):
        return {"model_id": "...", "revision": "..."}
```

The adapter must not use a separate inference engine or hidden/external memory
for a native run. `cache_metadata` is observational: the harness records it
after prefill and before asking the question. A retained/evicted result is only
reported when relevant position metadata is available.

## Run a diagnostic experiment

```python
from deeptrace import evaluate, generate_needle_cases, write_reports

cases = generate_needle_cases(
    tokenizer,
    distances=[1024, 4096],
    seed=42,
    cases_per_condition=1,
    include_negative_controls=True,
)
result = evaluate(
    zeta_adapter,
    tokenizer,
    cases,
    seed=42,
    dataset_id="zeta-needle-smoke-v1",
    tokenizer_metadata={"tokenizer_id": "...", "revision": "..."},
)
write_reports(result, "deeptrace-results")
```

Generated cases include early, middle, and late insertion positions by default.
Token distances are measured with `Tokenizer.encode`; a tokenizer implementation
should be deterministic for these counts to be reproducible. Large distances
can require substantial memory and runtime, so begin with small diagnostic
values. This library does not automatically launch experiments.

Every output directory contains `report.md`, `results.json`, `questions.jsonl`,
and `summary.csv`. Results include model/tokenizer metadata supplied by the
adapter and caller, timestamps, per-question answers, exact-match outcomes,
latency, and available cache observations. Add commit/checkpoint provenance to
the metadata supplied by the caller when available.

## Tests

The tests use only the Python standard library:

```sh
python -m unittest discover -s tests -v
```
