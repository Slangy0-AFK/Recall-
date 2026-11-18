# Environment and Dependency Report

- **Experiment ID:** `recall-qualification-e77f8c08-e2ba-4d53-bbcc-962040c1a6b1`
- **Timestamp:** 2026-11-18T13:21:31.624056+00:00 to 2026-11-18T13:21:34.158452+00:00
- **Recall commit:** `1c8cbf5bd997993ed8320cb2be568230aa8b4dd1`
- **Connected commits:** zeta-scaling-toolkit=`67d1a34a1ac35d9ea7894ffd9354d5e4e5b7d36b`; mathmesh-governor=`4ef77424b7ef4c5b131f558e85fa1eae24d890ab`; System-Stability-Toolkit=`9e7a254ee9607b23ae35b756110d3713ad0dac45`; deterministic-rag-governor=`7eee42bc77757e8a23138b9b65c8a1072c88ca9d`
- **Model provenance:** `HuggingFaceTB/SmolLM2-360M-Instruct` @ `a10cc1512eabd3dde888204e902eca88bddb4951`
- **Tokenizer provenance:** NOT LOADED (The benchmark's tokenizer was not instantiated in this run.)
- **Workload / actual token count:** One Zeta attention/cache component probe with sequence_length=32; no model or tokenizer was run. (actual model tokens: 0)
- **Hardware/runtime:** Linux-6.8.0-1070-azure-x86_64-with-glibc2.39; Python 3.14.2; torch 2.15.0+cu132; Node v24.22.2
- **Test status:** BLOCKED

- Environment status: PARTIAL
- Python: 3.14.2
- Installed versions: `{"fastapi": null, "psutil": "7.2.2", "pydantic": "2.14.3", "python-dotenv": null, "torch": "2.15.0", "uvicorn": null, "zeta-scaling-toolkit": "0.5.1"}`
- Runtime checks: `{"node": {"expected": ">=22.6 for direct TypeScript module loading", "observed": "v24.22.2", "passed": true}, "psutil": {"expected": "7.2.2", "observed": "7.2.2", "passed": true}, "python_minimum": {"expected": "3.10", "observed": "3.14.2", "passed": true}, "torch": {"expected": "2.15.0", "observed": "2.15.0", "passed": true}, "zeta-scaling-toolkit": {"expected": "0.5.1", "module_importable_without_checkout": false, "observed_distribution": "0.5.1", "observed_source_revision": "67d1a34a1ac35d9ea7894ffd9354d5e4e5b7d36b", "observed_source_version": "0.5.1", "passed": true}}`
- RAG pinned packages missing/mismatched: `["fastapi==0.115.6 (installed: missing)", "uvicorn==0.34.0 (installed: missing)", "pydantic==2.10.6 (installed: 2.14.3)", "python-dotenv==1.0.1 (installed: missing)"]`
- Recall requirements are pinned in `requirements.txt`; no packages were installed or upgraded during this run.
- MathMesh requires Node.js >=22 and PostgreSQL for its API. The pure parity-check module was run directly; no database-backed service was started.
- System-Stability-Toolkit and MathMesh are consumed from separately cloned source checkouts. RAG health is accessed over HTTP and its service is not configured.
- CUDA availability and system resource measurements are recorded in the JSON.
- Model weights for the pinned SmolLM2 revision are cached; no download or model load was attempted.
