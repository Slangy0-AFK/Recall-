# Four-System Integration Report

- **Experiment ID:** `recall-qualification-e77f8c08-e2ba-4d53-bbcc-962040c1a6b1`
- **Timestamp:** 2026-11-18T13:21:31.624056+00:00 to 2026-11-18T13:21:34.158452+00:00
- **Recall commit:** `1c8cbf5bd997993ed8320cb2be568230aa8b4dd1`
- **Connected commits:** zeta-scaling-toolkit=`67d1a34a1ac35d9ea7894ffd9354d5e4e5b7d36b`; mathmesh-governor=`4ef77424b7ef4c5b131f558e85fa1eae24d890ab`; System-Stability-Toolkit=`9e7a254ee9607b23ae35b756110d3713ad0dac45`; deterministic-rag-governor=`7eee42bc77757e8a23138b9b65c8a1072c88ca9d`
- **Model provenance:** `HuggingFaceTB/SmolLM2-360M-Instruct` @ `a10cc1512eabd3dde888204e902eca88bddb4951`
- **Tokenizer provenance:** NOT LOADED (The benchmark's tokenizer was not instantiated in this run.)
- **Workload / actual token count:** One Zeta attention/cache component probe with sequence_length=32; no model or tokenizer was run. (actual model tokens: 0)
- **Hardware/runtime:** Linux-6.8.0-1070-azure-x86_64-with-glibc2.39; Python 3.14.2; torch 2.15.0+cu132; Node v24.22.2
- **Test status:** BLOCKED

## Integration map

- **Zeta Scaling Toolkit:** `zeta_scaling_toolkit.NonCollapsingAttention` and `StreamingKVCache`; component probe only. No Hugging Face inference runner is present in the connected repository.
- **MathMesh Governor:** `governor/shared/mathChecks.ts:evaluateVoteParity`; actual parity validation. The app's `GET /api/health` and authenticated `POST /api/functions/recordAudit` are supported, but no deployment/token was configured and remote writes are opt-in.
- **System-Stability-Toolkit:** documented `crash_fix.py <scale> --json`; recommendations are recorded but not applied. The toolkit does not monitor resources; snapshots here are Recall/psutil measurements.
- **Deterministic RAG Governor:** documented `GET /health`; no running service was configured. The repo documents a governed `/govern` path, not a metadata-only audit mode. No retrieval or generation request was issued.

## Results

- Zeta: PASS
- MathMesh: PARTIAL
- System-Stability-Toolkit: PASS
- Deterministic RAG Governor: BLOCKED
- Four-system integration: BLOCKED
- Shared native telemetry events received: 0
- Event order is serial; no parallel execution is inferred from timestamps.
- Zeta probe output shape: `[1, 2, 32, 16]`; finite: `True`; cache positions: `[0, 3, 4]`; model tokens: `0`.
- MathMesh numeric observation: `{'aligned': True, 'reason': None}`; unsupported operations: `['radix conversion', 'numeric round-trip']`.
- MathMesh event delivery: BLOCKED; attempted 0, delivered 0, failed 0, dropped None.
- Stability snapshots (Recall/psutil): baseline 2026-11-18T13:21:31.654818+00:00: memory 46.3%, CPU 0.0%, process RSS 28.0 MB; during 2026-11-18T13:21:33.956795+00:00: memory 49.6%, CPU 0.0%, process RSS 579.4 MB; post 2026-11-18T13:21:34.057327+00:00: memory 49.6%, CPU 0.0%, process RSS 579.4 MB.
- RAG queries/audit events: `0`/`0`; event timestamps and evidence references: `[]` / `[]`; audit-only mode: `UNSUPPORTED`.

## Unsupported interfaces

- Zeta model-token inference for the pinned Hugging Face model.
- MathMesh radix conversion and numeric round-trip operations.
- MathMesh live governance/audit delivery without a dedicated deployment and admin token.
- A metadata-only RAG audit mode; `/govern` can retrieve and generate.
