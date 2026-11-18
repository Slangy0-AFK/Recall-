from __future__ import annotations

import importlib.metadata
import inspect
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import zeta_scaling_toolkit as zeta


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPECTED_VERSION = "0.5.1"
EXPECTED_COMMIT = "67d1a34a1ac35d9ea7894ffd9354d5e4e5b7d36b"
HIDDEN = 960
QUERY_HEADS = 15
KV_HEADS = 5
HEAD_DIM = 64
GROUP_SIZE = 3
BATCH = 2
SEQUENCE = 4
ATOL = 1e-5
RTOL = 1e-4


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def write_json(name: str, data: dict[str, Any]) -> None:
    (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    key_by_query_head = key.repeat_interleave(GROUP_SIZE, dim=2)
    value_by_query_head = value.repeat_interleave(GROUP_SIZE, dim=2)

    q = query.permute(0, 2, 1, 3)
    k = key_by_query_head.permute(0, 2, 1, 3)
    v = value_by_query_head.permute(0, 2, 1, 3)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
    if causal:
        future = torch.triu(
            torch.ones((SEQUENCE, SEQUENCE), dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(future, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    output = probabilities @ v
    return output.permute(0, 2, 1, 3).reshape(BATCH, SEQUENCE, HIDDEN)


def main() -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    root = git("rev-parse", "--show-toplevel")
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")

    distribution = importlib.metadata.distribution("zeta-scaling-toolkit")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    source_root = Path(zeta.__file__).resolve().parent
    source_files = sorted(source_root.glob("*.py"))
    source_text = "\n".join(path.read_text() for path in source_files)
    gqa_symbols = [
        name
        for name in dir(zeta)
        if any(term in name.lower() for term in ("gqa", "grouped", "multiquery"))
    ]

    attention = zeta.NonCollapsingAttention(d_k=HEAD_DIM, check_finite=True).eval()
    api_inspection = {
        "recorded_at_utc": timestamp,
        "repository": {"root": root, "branch": branch, "head": head},
        "dependency": {
            "distribution": distribution.metadata["Name"],
            "version": distribution.version,
            "module": zeta.__name__,
            "module_path": str(Path(zeta.__file__).resolve()),
            "source_root": str(source_root),
            "source_files": [path.name for path in source_files],
            "direct_url": direct_url,
            "expected_version": EXPECTED_VERSION,
            "expected_commit": EXPECTED_COMMIT,
            "pinned_version_matches": distribution.version == EXPECTED_VERSION,
            "pinned_commit_matches": direct_url.get("vcs_info", {}).get("commit_id")
            == EXPECTED_COMMIT,
        },
        "attention_apis": {
            "public_exports": [
                name for name in zeta.__all__ if "Attention" in name or "attention" in name
            ],
            "non_collapsing_attention": {
                "import_path": "zeta_scaling_toolkit.NonCollapsingAttention",
                "class": "NonCollapsingAttention",
                "constructor_signature": str(inspect.signature(zeta.NonCollapsingAttention)),
                "forward_signature": str(inspect.signature(zeta.NonCollapsingAttention.forward)),
                "source": str(Path(inspect.getsourcefile(zeta.NonCollapsingAttention)).resolve()),
                "source_method": "NonCollapsingAttention.forward",
                "supported_head_dimension": "d_k is constructor-selected; forward requires Q/K last dimension equal d_k; 64 accepted",
                "input_layout": "[..., query_or_key_sequence, d_k]; head axes are leading batch dimensions",
                "grouped_query_heads_direct": False,
                "gqa_specific_exports": gqa_symbols,
                "gqa_symbol_search_in_source": "none"
                if not any(term in source_text.lower() for term in ("grouped-query", "grouped_query", "gqa"))
                else "matching source text found; inspect exported API",
                "standard_scaled_dot_product": False,
                "causal_support": "attn_mask/visibility_mask arguments; boolean True means visible/allowed; no dedicated is_causal argument",
                "normalizes_q_and_k": "_unit(q.to(dtype)) and _unit(k.to(dtype)) in forward",
                "learned_temperature": "bounded self.tau used for cosine logits",
                "source_locations": {
                    "public_exports": "zeta_scaling_toolkit/__init__.py:1-21",
                    "class": "zeta_scaling_toolkit/core.py:56",
                    "constructor": "NonCollapsingAttention.__init__ at core.py:78",
                    "forward": "NonCollapsingAttention.forward at core.py:148",
                    "qk_unit_normalization": "core.py:200-201",
                    "cosine_temperature_path": "core.py:236-277",
                },
            },
            "module_inventory": {
                "classes": [
                    name
                    for name, value in vars(zeta.core).items()
                    if inspect.isclass(value) and value.__module__ == zeta.core.__name__
                ],
                "gqa_specific_implementation_found": False,
            },
        },
        "model_or_tokenizer_loaded": False,
        "checkpoint_tensor_values_loaded": False,
        "inference_run": False,
    }

    generator = torch.Generator(device="cpu").manual_seed(20261118)
    query_input = torch.randn(
        (BATCH, SEQUENCE, QUERY_HEADS, HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
    )
    key_input = torch.randn(
        (BATCH, SEQUENCE, KV_HEADS, HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
    )
    value_input = torch.randn(
        (BATCH, SEQUENCE, KV_HEADS, HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
    )

    # The checkpoint's [B,S,H,D] layout is accepted positionally but Zeta
    # interprets the penultimate dimension as sequence, so record the trap.
    direct_layout_output = attention(query_input, key_input, value_input)

    query = query_input.permute(0, 2, 1, 3).detach().requires_grad_(True)
    key = key_input.permute(0, 2, 1, 3).detach().requires_grad_(True)
    value = value_input.permute(0, 2, 1, 3).detach().requires_grad_(True)
    causal_mask = torch.tril(torch.ones((SEQUENCE, SEQUENCE), dtype=torch.bool))

    direct_gqa_rejected = False
    direct_gqa_error = None
    try:
        attention(query, key, value, attn_mask=causal_mask)
    except ValueError as error:
        direct_gqa_rejected = True
        direct_gqa_error = str(error)

    expanded_key = key.repeat_interleave(GROUP_SIZE, dim=1)
    expanded_value = value.repeat_interleave(GROUP_SIZE, dim=1)
    candidate_output = attention(
        query,
        expanded_key,
        expanded_value,
        attn_mask=causal_mask,
    )
    candidate_flat = candidate_output.permute(0, 2, 1, 3).reshape(
        BATCH, SEQUENCE, HIDDEN
    )
    reference = reference_attention(query_input, key_input, value_input, causal=True)
    max_abs_error = float((candidate_flat.detach() - reference).abs().max())
    mean_abs_error = float((candidate_flat.detach() - reference).abs().mean())
    numerically_equivalent = torch.allclose(
        candidate_flat.detach(), reference, atol=ATOL, rtol=RTOL
    )

    candidate_flat.square().mean().backward()
    gradients_finite = all(
        tensor.grad is not None and bool(torch.isfinite(tensor.grad).all())
        for tensor in (query, key, value)
    )

    changed_key = key.detach().clone()
    changed_value = value.detach().clone()
    changed_key[:, :, 2:, :] += 9.0
    changed_value[:, :, 2:, :] -= 7.0
    changed_output = attention(
        query.detach(),
        changed_key.repeat_interleave(GROUP_SIZE, dim=1),
        changed_value.repeat_interleave(GROUP_SIZE, dim=1),
        attn_mask=causal_mask,
    )
    candidate_causal_prefix_unchanged = torch.allclose(
        candidate_output.detach()[:, :, :2, :],
        changed_output[:, :, :2, :],
        atol=ATOL,
        rtol=RTOL,
    )

    expected_mapping = [head // GROUP_SIZE for head in range(QUERY_HEADS)]
    mapping_is_3_to_1 = expected_mapping == [
        kv_head
        for kv_head in range(KV_HEADS)
        for _ in range(GROUP_SIZE)
    ]
    synthetic_reference = reference_attention(
        query_input, key_input, value_input, causal=True
    )
    reference_tests = {
        "status": "PASS",
        "implementation": "Independent PyTorch matmul, explicit K/V repeat_interleave, 1/sqrt(64) scaling, triangular causal mask, softmax",
        "group_mapping_query_head_to_kv_head": expected_mapping,
        "grouping_3_to_1_verified": mapping_is_3_to_1,
        "input_shapes": {
            "query": list(query_input.shape),
            "key": list(key_input.shape),
            "value": list(value_input.shape),
        },
        "output_shape": list(synthetic_reference.shape),
        "output_shape_expected": [BATCH, SEQUENCE, HIDDEN],
        "finite_output": bool(torch.isfinite(synthetic_reference).all()),
        "causal_masking": "Future K/V perturbations do not affect earlier reference outputs",
        "reference_uses_zeta_attention": False,
        "model_or_checkpoint_used": False,
    }

    changed_ref = reference_attention(
        query_input,
        torch.cat((key_input[:, :2], key_input[:, 2:] + 9.0), dim=1),
        torch.cat((value_input[:, :2], value_input[:, 2:] - 7.0), dim=1),
        causal=True,
    )
    reference_tests["causal_prefix_invariance_verified"] = torch.allclose(
        synthetic_reference[:, :2], changed_ref[:, :2], atol=ATOL, rtol=RTOL
    )

    gqa_tests = {
        "recorded_at_utc": timestamp,
        "decision": "FAIL",
        "decision_reason": "The only exported attention implementation tested rejects native 15:5 GQA shapes and, after explicit KV expansion, fails standard scaled-dot-product numerical equivalence.",
        "contract": {
            "device": "CPU",
            "dtype": "FP32",
            "hidden_dimension": HIDDEN,
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "head_dimension": HEAD_DIM,
            "grouping": "3 query heads per KV head",
            "raw_qk_dot_products": "FAIL: source normalizes Q and K before logits",
            "scaled_scores_1_over_sqrt_d": "FAIL: source uses bounded learned tau over cosine similarity",
            "causal_masking": "PASS when a lower-triangular boolean attn_mask is supplied (True means visible/allowed)",
            "output_dimensions": {
                "candidate_internal": list(candidate_output.shape),
                "flattened": list(candidate_flat.shape),
                "expected": [BATCH, SEQUENCE, HIDDEN],
                "status": "PASS after explicit KV head expansion and head concatenation",
            },
            "finite_outputs": bool(torch.isfinite(candidate_output).all()),
            "finite_gradients": gradients_finite,
        },
        "observations": {
            "checkpoint_layout_direct_call_output_shape": list(direct_layout_output.shape),
            "checkpoint_layout_direct_call_semantics": "Shape coincides, but Zeta interprets the penultimate axis as sequence; this call attends across the 15/5 head axes independently at each original sequence position.",
            "native_gqa_call_rejected": direct_gqa_rejected,
            "native_gqa_error": direct_gqa_error,
            "explicit_kv_expansion_factor": GROUP_SIZE,
            "expanded_candidate_output_shape": list(candidate_flat.shape),
            "candidate_causal_prefix_invariance": candidate_causal_prefix_unchanged,
            "numerically_equivalent_to_reference": numerically_equivalent,
            "comparison_atol": ATOL,
            "comparison_rtol": RTOL,
            "maximum_absolute_error": max_abs_error,
            "mean_absolute_error": mean_abs_error,
        },
        "safety": {
            "synthetic_tensors_only": True,
            "model_loaded": False,
            "tokenizer_loaded": False,
            "checkpoint_values_read_or_loaded": False,
            "inference_run": False,
            "memory_gate_mib": 6144,
            "memory_gate_changed": False,
        },
    }

    from adapters.zeta_adapter import ZetaToolkitAdapter

    adapter = ZetaToolkitAdapter(
        repository=None,
        expected_version=EXPECTED_VERSION,
        sequence_length=SEQUENCE,
        head_count=QUERY_HEADS,
        head_dimension=HEAD_DIM,
    )
    initialization = adapter.initialize()
    capabilities = adapter.capabilities()
    adapter_readiness = {
        "recorded_at_utc": timestamp,
        "recall_adapter_import": "PASS",
        "adapter_initialization": initialization,
        "capabilities": capabilities,
        "can_invoke_compatible_gqa": False,
        "reason": "The adapter invokes NonCollapsingAttention with identical Q/K/V head_count and does not expose a GQA-specific operation or standard scaled-dot-product semantics.",
        "model_runner_exposed": capabilities["model_runner"],
        "tokenizer_exposed": capabilities["tokenizer"],
        "model_loaded": False,
        "tokenizer_loaded": False,
        "inference_run": False,
        "memory_gate_mib": 6144,
        "memory_gate_changed": False,
    }

    assert distribution.version == EXPECTED_VERSION
    assert direct_url.get("vcs_info", {}).get("commit_id") == EXPECTED_COMMIT
    assert direct_gqa_rejected
    assert mapping_is_3_to_1
    assert candidate_causal_prefix_unchanged
    assert gradients_finite
    assert reference_tests["causal_prefix_invariance_verified"]
    assert not numerically_equivalent
    assert list(candidate_flat.shape) == [BATCH, SEQUENCE, HIDDEN]

    write_json("ZETA_API_INSPECTION.json", api_inspection)
    write_json("GQA_TEST_RESULTS.json", gqa_tests)
    write_json("REFERENCE_COMPARISON.json", {
        **reference_tests,
        "zeta_candidate": {
            "maximum_absolute_error": max_abs_error,
            "mean_absolute_error": mean_abs_error,
            "allclose_at_tolerance": numerically_equivalent,
            "tolerance": {"atol": ATOL, "rtol": RTOL},
        },
    })
    write_json("RECALL_ADAPTER_READINESS.json", adapter_readiness)
    print(json.dumps({
        "zeta_acceptance": gqa_tests["decision"],
        "native_gqa_rejected": direct_gqa_rejected,
        "max_abs_error": max_abs_error,
        "finite_gradients": gradients_finite,
        "causal_prefix_invariance": candidate_causal_prefix_unchanged,
        "adapter_ready_for_compatible_gqa": False,
    }, indent=2))


if __name__ == "__main__":
    main()
