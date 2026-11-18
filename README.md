# Recall Qualification Harness

Recall coordinates evidence collection for the four infrastructure repositories.
It does not replace their APIs or claim that a component is operational without
calling its real interface. The first run records a qualification result and
explicitly reports unsupported or unconfigured integration paths.

## Requirements

- Python 3.10+
- The exact Python dependencies pinned in [`requirements.txt`](requirements.txt)
- Node.js 22.6+ for MathMesh's `evaluateVoteParity` TypeScript module
- Local source checkouts of Zeta, MathMesh, and System-Stability-Toolkit at the
  revisions pinned in [`config/experiment.json`](config/experiment.json)
- A running RAG Governor service is optional for its read-only health probe.
  Recall will not call its `/govern` endpoint as that may retrieve and generate.

Install the pinned Python dependencies in a dedicated environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The safe suite also includes the legacy pytest-based DeepTrace tests. Install
the pinned test runner with:

```bash
python -m pip install -r requirements-dev.txt
```

No dependency installation is performed by the harness. MathMesh's live API
additionally requires its documented Node/PostgreSQL deployment. The RAG
service's separately pinned requirements are in that repository; Recall does
not install them.

## Configure and run

Point Recall at checkouts of the pinned repositories:

```bash
export RECALL_ZETA_REPO=/path/to/zeta-scaling-toolkit
export RECALL_MATHMESH_REPO=/path/to/mathmesh-governor
export RECALL_STABILITY_REPO=/path/to/System-Stability-Toolkit
python recall_harness.py
```

Optional read-only service probes use `RECALL_MATHMESH_BASE_URL` and
`RECALL_RAG_BASE_URL`. Remote MathMesh audit writes require both
`RECALL_MATHMESH_ADMIN_TOKEN` and the explicit opt-in
`RECALL_ALLOW_REMOTE_AUDIT_WRITES=1`; the harness does not enable that opt-in
by default. A RAG `/govern` request can retrieve, generate, spend agent credit,
and write audit events. It is disabled unless `RECALL_ALLOW_RAG_GOVERN=1`,
`RECALL_RAG_AGENT_ID`, and `RECALL_RAG_QUERY` are all explicitly configured.
Recall never changes Zeta inference settings, applies stability
recommendations, downloads model weights, or starts a long benchmark
automatically.

The single safe qualification command runs full unittest discovery and then
the non-destructive adapter probes. Select a new output directory for each run;
Recall never overwrites existing result JSON or reports:

```bash
python recall_harness.py --safe-suite --output-dir evidence/qualification-run-001
```

The command's process exit status remains nonzero if the overall qualification
gate is blocked, even when the unit tests themselves pass. For a no-write safe
run, append `--no-write`. For individual operations:

```bash
python recall_harness.py --system zeta_scaling_toolkit --no-write
python recall_harness.py --integration-only
python recall_harness.py --fault-isolation
```

Integration-only runs perform the test suite's safe contract checks and
configured read-only health probes. Fault-injection checks exercise Recall's
local adapter boundaries; they are not evidence of live service resilience.
Real adapter tests need the corresponding source checkout variables configured.

Validate saved evidence without rerunning anything, or regenerate Markdown
reports from existing evidence. Report regeneration never writes or replaces
the JSON evidence; replacing existing Markdown files requires an explicit flag:

```bash
python recall_harness.py --validate-evidence results/RECALL_TEST_RESULTS.json
python recall_harness.py --from-evidence results/RECALL_TEST_RESULTS.json --overwrite-reports
```

The default evidence filenames in `results/` and `reports/` are created only
when absent. Use `--output-dir` to retain each subsequent run separately.

## Integration map and evidence limits

| Repository | Interface Recall invokes | What it verifies |
|---|---|---|
| Zeta Scaling Toolkit | `zeta_scaling_toolkit.NonCollapsingAttention` and `StreamingKVCache` | Finite attention output and real cache positions. The repository does not expose a Hugging Face model runner. |
| MathMesh Governor | `governor/shared/mathChecks.ts:evaluateVoteParity`; optionally `GET /api/health` and authenticated `POST /api/functions/recordAudit` | Actual parity checks. The present repository does not provide base conversion or numeric round-trip operations. Audit writes are opt-in and require a deployed app. |
| System-Stability-Toolkit | Documented `crash_fix.py <scale> --json` CLI | Advisory limits only. Recall separately captures resource snapshots using `psutil`; the toolkit itself does not monitor memory/CPU or measure temperature. |
| Deterministic RAG Governor | `GET /health`, `GET /ready`; opt-in `POST /govern` | The governed operation combines retrieval, generation, spend, and audit writes. There is no metadata-only audit endpoint or separate retrieval/generation API. |

Recall owns the experiment ID, ordered local observations, resource snapshots,
and report generation. These observations are not component-originated audit
events. No overlapping timestamps are treated as proof of parallel execution.
Unsupported operations are listed separately from tested failures.

The connected Zeta repository contains attention/cache primitives but no
supported adapter that runs the pinned
`HuggingFaceTB/SmolLM2-360M-Instruct` revision through Zeta. The harness will
therefore block its short-inference and performance-comparison phases rather
than substitute plain Hugging Face inference. A completed 100,000-token
workload is a total processed-token count, not 100,000 tokens of simultaneous
attention or retained context.

Statuses are `PASS`, `FAIL`, `PARTIAL`, `BLOCKED`, `NOT TESTED`, and
`NOT APPLICABLE`. A system check can pass while the overall integration remains
blocked; missing evidence is never treated as success. A full integration
requires every pinned system revision, all required adapter checks, and native
events correlated to one experiment ID. The connected systems do not currently
expose that shared event contract.

## Adapter contract

Each adapter exposes `initialize()`, `capabilities()`, `health()`,
`execute(operation, input_metadata)`, and `shutdown()`. The harness
normalizes each operation into a record containing experiment ID, system and
component, timestamp, operation, status, input/output metadata, actual
workload, elapsed time, resource measurements, error details, repository
revision, and software version. Adapter errors are recorded as `BLOCKED` for
missing prerequisites and `FAIL` for an executed operation that fails.
An adapter may return `native_events`; each event must carry the shared
experiment ID, the matching system name, component, timestamp, operation, and
an allowed status. Recall only counts such validated component-originated
events toward the four-system gate.

`config/capabilities.json` is the capability declaration source;
`config/integrations.json` describes cross-system gates and write-side-effect
opt-ins. In particular, Zeta's adapter consumes only its public
`NonCollapsingAttention` and `StreamingKVCache` primitives. It does not import
private internals or impersonate the missing Hugging Face model runner. The
future Zeta runner must implement the public `ZetaInferenceRunner` protocol
defined in [`adapters/zeta_adapter.py`](adapters/zeta_adapter.py):

```python
initialize(model_id, revision, precision) -> ZetaModelProvenance
capabilities() -> Mapping[str, bool]
health() -> Mapping[str, object]
infer(input_tokens) -> ZetaInferenceResult
shutdown() -> None
```

`ZetaModelProvenance` records repository commit, software version, exact model
and tokenizer revisions, and precision. `ZetaInferenceResult` records input
and actually processed tokens, elapsed time, throughput, numerical validation,
cache positions/next position, and output metadata. Missing runner support
raises `AdapterBlocked`; numerical validation failures must be explicit, and
other execution errors must use `ZetaRunnerError` or propagate without being
converted into success-shaped results.

## Evidence outputs

- [`results/RECALL_TEST_RESULTS.json`](results/RECALL_TEST_RESULTS.json)
- [`reports/RECALL_TEST_RESULTS.md`](reports/RECALL_TEST_RESULTS.md)
- [`reports/FOUR_SYSTEM_INTEGRATION_REPORT.md`](reports/FOUR_SYSTEM_INTEGRATION_REPORT.md)
- [`reports/FAULT_ISOLATION_REPORT.md`](reports/FAULT_ISOLATION_REPORT.md)
- [`reports/PERFORMANCE_COMPARISON.md`](reports/PERFORMANCE_COMPARISON.md)
- [`reports/ENVIRONMENT_AND_DEPENDENCY_REPORT.md`](reports/ENVIRONMENT_AND_DEPENDENCY_REPORT.md)
- [`reports/RECALL_COMPLETION_REPORT.md`](reports/RECALL_COMPLETION_REPORT.md)

Each generated report records the experiment ID, timestamp, repository
revisions, model provenance, workload/token count, runtime details, and status.
The user-provided 10.7199 tokens/second reference is not treated as verified by
Recall; 21.4398 tokens/second is a measurement target, not an assumed result.
