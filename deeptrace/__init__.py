"""Zeta DeepTrace: Native long-distance retrieval testing framework.

Built directly on the Zeta Scaling Toolkit's mathematical foundation.
"""

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
]
