# Performance Comparison

- **Experiment ID:** `recall-qualification-e77f8c08-e2ba-4d53-bbcc-962040c1a6b1`
- **Timestamp:** 2026-11-18T13:21:31.624056+00:00 to 2026-11-18T13:21:34.158452+00:00
- **Recall commit:** `1c8cbf5bd997993ed8320cb2be568230aa8b4dd1`
- **Connected commits:** zeta-scaling-toolkit=`67d1a34a1ac35d9ea7894ffd9354d5e4e5b7d36b`; mathmesh-governor=`4ef77424b7ef4c5b131f558e85fa1eae24d890ab`; System-Stability-Toolkit=`9e7a254ee9607b23ae35b756110d3713ad0dac45`; deterministic-rag-governor=`7eee42bc77757e8a23138b9b65c8a1072c88ca9d`
- **Model provenance:** `HuggingFaceTB/SmolLM2-360M-Instruct` @ `a10cc1512eabd3dde888204e902eca88bddb4951`
- **Tokenizer provenance:** NOT LOADED (The benchmark's tokenizer was not instantiated in this run.)
- **Workload / actual token count:** One Zeta attention/cache component probe with sequence_length=32; no model or tokenizer was run. (actual model tokens: 0)
- **Hardware/runtime:** Linux-6.8.0-1070-azure-x86_64-with-glibc2.39; Python 3.14.2; torch 2.15.0+cu132; Node v24.22.2
- **Test status:** BLOCKED

## Historical reference (provided by the user)

- Model/revision: `HuggingFaceTB/SmolLM2-360M-Instruct` @ `a10cc1512eabd3dde888204e902eca88bddb4951`
- Workload: 100,000 contiguous processed tokens; native context 8,192
- Reference throughput: 10.7199 tokens/s
- Reference wall time: 9636.7 s
- Reference final token position: 99999
- Checkpoints: 56/56 passed
- Reference evidence was not locally verified. A total processed-token count is not simultaneous attention/context retention.

## Current result

- Short inference: BLOCKED; no model inference was run.
- Actual processed model tokens: 0
- Measured inference time: none
- Pinned model weights cached locally: True
- Workload throughput formula: actual processed model tokens / measured inference time; not calculable for this probe.
- Total end-to-end Recall harness elapsed time is recorded in `results/RECALL_TEST_RESULTS.json`; it is not model inference time.
- 21.4398 tokens/s target: NOT MEASURED; achievement is unverified.
- Comparability: not applicable. No inference run exists to compare for model, tokenizer, workload, hardware, precision, or measurement method.
