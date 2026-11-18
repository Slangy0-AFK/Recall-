# Fault Isolation Report

- **Experiment ID:** `recall-qualification-e77f8c08-e2ba-4d53-bbcc-962040c1a6b1`
- **Timestamp:** 2026-11-18T13:21:31.624056+00:00 to 2026-11-18T13:21:34.158452+00:00
- **Recall commit:** `1c8cbf5bd997993ed8320cb2be568230aa8b4dd1`
- **Connected commits:** zeta-scaling-toolkit=`67d1a34a1ac35d9ea7894ffd9354d5e4e5b7d36b`; mathmesh-governor=`4ef77424b7ef4c5b131f558e85fa1eae24d890ab`; System-Stability-Toolkit=`9e7a254ee9607b23ae35b756110d3713ad0dac45`; deterministic-rag-governor=`7eee42bc77757e8a23138b9b65c8a1072c88ca9d`
- **Model provenance:** `HuggingFaceTB/SmolLM2-360M-Instruct` @ `a10cc1512eabd3dde888204e902eca88bddb4951`
- **Tokenizer provenance:** NOT LOADED (The benchmark's tokenizer was not instantiated in this run.)
- **Workload / actual token count:** One Zeta attention/cache component probe with sequence_length=32; no model or tokenizer was run. (actual model tokens: 0)
- **Hardware/runtime:** Linux-6.8.0-1070-azure-x86_64-with-glibc2.39; Python 3.14.2; torch 2.15.0+cu132; Node v24.22.2
- **Test status:** BLOCKED

**Fault-isolation status:** PARTIAL

The harness isolates adapter outcomes and continues collecting evidence from available components when another adapter is blocked. No mock or simulated result is counted as a real integration.

### Observed blockers

- deterministic_rag_governor: No RAG service URL is configured. The repository has no documented metadata-only audit endpoint; /govern may retrieve and generate.
- MathMesh audit writes were not attempted because no explicit remote-write authorization and staging endpoint were configured.
- RAG audit-only execution was not attempted because the documented API has no audit-only mode and `/govern` can trigger retrieval/generation.

### Limits

A live outage test against all four services was not possible: MathMesh and RAG deployments were not configured. This is not a production-readiness test.
