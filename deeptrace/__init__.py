"""Zeta DeepTrace: Native long-distance retrieval testing framework.

Built directly on the Zeta Scaling Toolkit's mathematical foundation.
"""

from typing import Any

from .fact_generator import FactGenerator, Fact, generate_distractors
from .retrieval_dataset import RetrievalDataset, TokenPosition
from .retrieval_tests import (
    RetrievalResult,
    RetrievalScorer,
    TestA_SingleFactRetrieval,
    TestB_MultipleFactRetrieval,
    TestC_MultiHopRetrieval,
    TestD_AggregationRetrieval,
    TestE_NegativeRetrieval,
)
from .experiment import DeepTraceConfig, DeepTraceExperiment
from .tokenizer import SimpleTokenizer
from .core import (
    build_negative_case,
    build_needle_case,
    build_positive_control_case,
    evaluate,
    generate_needle_cases,
)
from .reporting import write_reports


def __getattr__(name: str) -> Any:
    if name in {"ZetaLanguageModel", "ZetaTransformerBlock"}:
        from . import zeta_inference

        return getattr(zeta_inference, name)
    if name == "NativeZetaIntegration":
        from .native_integration import NativeZetaIntegration

        return NativeZetaIntegration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "0.1.0"
__all__ = [
    "FactGenerator",
    "Fact",
    "generate_distractors",
    "RetrievalDataset",
    "TokenPosition",
    "RetrievalResult",
    "RetrievalScorer",
    "TestA_SingleFactRetrieval",
    "TestB_MultipleFactRetrieval",
    "TestC_MultiHopRetrieval",
    "TestD_AggregationRetrieval",
    "TestE_NegativeRetrieval",
    "DeepTraceConfig",
    "DeepTraceExperiment",
    "build_negative_case",
    "build_needle_case",
    "build_positive_control_case",
    "evaluate",
    "generate_needle_cases",
    "write_reports",
    "ZetaLanguageModel",
    "ZetaTransformerBlock",
    "SimpleTokenizer",
    "NativeZetaIntegration",
]
