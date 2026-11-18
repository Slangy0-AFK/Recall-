# Recall Test Results

- **Experiment ID:** `recall-qualification-e77f8c08-e2ba-4d53-bbcc-962040c1a6b1`
- **Timestamp:** 2026-11-18T13:21:31.624056+00:00 to 2026-11-18T13:21:34.158452+00:00
- **Recall commit:** `1c8cbf5bd997993ed8320cb2be568230aa8b4dd1`
- **Connected commits:** zeta-scaling-toolkit=`67d1a34a1ac35d9ea7894ffd9354d5e4e5b7d36b`; mathmesh-governor=`4ef77424b7ef4c5b131f558e85fa1eae24d890ab`; System-Stability-Toolkit=`9e7a254ee9607b23ae35b756110d3713ad0dac45`; deterministic-rag-governor=`7eee42bc77757e8a23138b9b65c8a1072c88ca9d`
- **Model provenance:** `HuggingFaceTB/SmolLM2-360M-Instruct` @ `a10cc1512eabd3dde888204e902eca88bddb4951`
- **Tokenizer provenance:** NOT LOADED (The benchmark's tokenizer was not instantiated in this run.)
- **Workload / actual token count:** One Zeta attention/cache component probe with sequence_length=32; no model or tokenizer was run. (actual model tokens: 0)
- **Hardware/runtime:** Linux-6.8.0-1070-azure-x86_64-with-glibc2.39; Python 3.14.2; torch 2.15.0+cu132; Node v24.22.2
- **Test status:** BLOCKED

**Qualification status:** BLOCKED  
**21.4398 tokens/s target:** NOT MEASURED (`target_reached=None`)  
**Production readiness:** NOT ASSESSED

## Phase results

| Phase | Status |
|---|---|
| 1 dependency environment validation | PARTIAL |
| 2 individual adapter tests | PARTIAL |
| 3 four system lightweight integration | BLOCKED |
| 4 shared experiment id telemetry correlation | PARTIAL |
| 5 numerical and checkpoint integrity | PARTIAL |
| 6 component outage fault isolation | PARTIAL |
| 7 short zeta inference benchmark | BLOCKED |
| 8 long benchmark | NOT TESTED |

## Adapter results

- **zeta_scaling_toolkit: PASS** — The probe exercises toolkit attention/cache primitives only. The toolkit does not expose a Hugging Face model runner.
- **mathmesh_governor: PARTIAL** — This supported check validates parity; it does not implement base conversion or a round-trip operation.
- **system_stability_toolkit: PASS** — Advice is recorded only; no workload or resource limits are changed.
- **deterministic_rag_governor: BLOCKED** — No RAG service URL is configured. The repository has no documented metadata-only audit endpoint; /govern may retrieve and generate.

## Performance

- Actual model tokens: 0
- Model inference elapsed time: not measured
- Model workload throughput: not measured
- Total harness wall time: 2.534395 seconds
- Blocker: The Zeta repository exposes attention/cache primitives, not a runner for the pinned Hugging Face model. Pinned model weights are present in the local Hugging Face cache. No model was loaded or run; the qualification gates do not permit inference.

The attention probe's sequence length is not a token count. The historical 100,000-token run is a total workload and does not mean 100,000 tokens of simultaneous attention or context retention.
