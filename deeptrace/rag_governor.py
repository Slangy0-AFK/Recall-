"""Deterministic RAG Governor: Retrieval and audit workflow management.

Implements deterministic retrieval operations, audit logging, and clear
identification of retrieval context integration into the generation pipeline.
"""

import json
import time
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class RetrievalMode(Enum):
    """Retrieval operational mode."""
    NATIVE_ONLY = "native_only"  # No external retrieval, audit-only
    AUGMENTED = "augmented"       # External retrieval integrated into generation
    HYBRID = "hybrid"             # Mixed retrieval sources


class RetrievalSource(Enum):
    """Source of retrieved evidence."""
    NATIVE_CACHE = "native_cache"
    EXTERNAL_KB = "external_kb"
    NONE = "none"


@dataclass
class RetrievalQuery:
    """A retrieval query within the experiment."""
    query_id: str
    experiment_id: str
    timestamp: str
    query_text: str
    expected_evidence: Optional[str] = None
    retrieved: bool = False


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    query_id: str
    timestamp: str
    query_text: str
    retrieval_mode: str
    evidence_retrieved: bool
    evidence_text: Optional[str] = None
    evidence_source: str = "none"
    confidence_score: float = 0.0
    retrieval_time_ms: float = 0.0
    citation_id: Optional[str] = None


@dataclass
class AuditEvent:
    """Audit log event for retrieval and integration operations."""
    event_id: str
    experiment_id: str
    timestamp: str
    event_type: str  # "query", "retrieval", "integration", "audit"
    details: Dict[str, Any]


@dataclass
class DeterministicRAGState:
    """State tracking for Deterministic RAG Governor."""
    experiment_id: str
    retrieval_mode: str
    start_time: float
    queries_issued: int = 0
    queries_answered: int = 0
    retrieval_events: int = 0
    retrieval_successful: int = 0
    retrieval_failed: int = 0
    audit_events: List[AuditEvent] = field(default_factory=list)
    retrieval_results: List[RetrievalResult] = field(default_factory=list)
    integration_points: List[Dict[str, Any]] = field(default_factory=list)


class DeterministicRAGGovernor:
    """Governance and audit for retrieval and generation augmentation."""

    def __init__(
        self,
        experiment_id: str,
        retrieval_mode: RetrievalMode = RetrievalMode.NATIVE_ONLY,
        output_dir: Path = Path("deeptrace/results"),
        external_knowledge_base: Optional[Dict[str, str]] = None
    ):
        """Initialize Deterministic RAG Governor.

        Args:
            experiment_id: Unique identifier for the experiment
            retrieval_mode: Operational mode (native_only, augmented, hybrid)
            output_dir: Directory for audit and results output
            external_knowledge_base: Optional external KB for augmented mode
        """
        self.experiment_id = experiment_id
        self.retrieval_mode = retrieval_mode.value
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.state = DeterministicRAGState(
            experiment_id=experiment_id,
            retrieval_mode=self.retrieval_mode,
            start_time=time.time()
        )

        self.external_kb = external_knowledge_base or {}
        self.query_counter = 0

        logger.info(f"Deterministic RAG Governor initialized for experiment {experiment_id} in {self.retrieval_mode} mode")

    def issue_retrieval_query(self, query_text: str, expected_evidence: Optional[str] = None) -> RetrievalQuery:
        """Issue a deterministic retrieval query.

        Args:
            query_text: The query to retrieve evidence for
            expected_evidence: Optional ground truth evidence (for auditing)

        Returns:
            RetrievalQuery object with unique ID
        """
        self.query_counter += 1
        query_id = f"QUERY-{self.experiment_id}-{self.query_counter:04d}"

        query = RetrievalQuery(
            query_id=query_id,
            experiment_id=self.experiment_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            query_text=query_text,
            expected_evidence=expected_evidence,
            retrieved=False
        )

        self.state.queries_issued += 1

        # Audit the query issuance
        self._audit_event(
            event_type="query",
            details={
                "query_id": query_id,
                "query_text": query_text,
                "expected_evidence": expected_evidence,
                "retrieval_mode": self.retrieval_mode
            }
        )

        logger.info(f"Query issued: {query_id}")
        return query

    def execute_retrieval(self, query: RetrievalQuery) -> RetrievalResult:
        """Execute retrieval for a query in the configured mode.

        Args:
            query: RetrievalQuery to process

        Returns:
            RetrievalResult with retrieval outcome and evidence (if found)
        """
        start_time = time.time()
        self.state.retrieval_events += 1

        logger.info(f"Executing retrieval for {query.query_id} in {self.retrieval_mode} mode")

        # Retrieve based on configured mode
        if self.retrieval_mode == RetrievalMode.NATIVE_ONLY:
            # Audit-only: no actual retrieval, just record
            evidence_text = None
            evidence_source = RetrievalSource.NONE.value
            confidence = 0.0
            retrieved = False

        elif self.retrieval_mode == RetrievalMode.AUGMENTED:
            # Try external KB
            evidence_text, found = self._search_external_kb(query.query_text)
            evidence_source = RetrievalSource.EXTERNAL_KB.value if found else RetrievalSource.NONE.value
            confidence = 0.95 if found else 0.0
            retrieved = found
            if found:
                self.state.retrieval_successful += 1
            else:
                self.state.retrieval_failed += 1

        else:  # HYBRID
            # Try both
            native_evidence = self._search_native_cache(query.query_text)
            external_evidence, external_found = self._search_external_kb(query.query_text)

            if external_found:
                evidence_text = external_evidence
                evidence_source = RetrievalSource.EXTERNAL_KB.value
                confidence = 0.95
                retrieved = True
                self.state.retrieval_successful += 1
            elif native_evidence:
                evidence_text = native_evidence
                evidence_source = RetrievalSource.NATIVE_CACHE.value
                confidence = 0.85
                retrieved = True
                self.state.retrieval_successful += 1
            else:
                evidence_text = None
                evidence_source = RetrievalSource.NONE.value
                confidence = 0.0
                retrieved = False
                self.state.retrieval_failed += 1

        retrieval_time_ms = (time.time() - start_time) * 1000
        citation_id = f"CITE-{self.query_counter:04d}" if retrieved else None

        result = RetrievalResult(
            query_id=query.query_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            query_text=query.query_text,
            retrieval_mode=self.retrieval_mode,
            evidence_retrieved=retrieved,
            evidence_text=evidence_text,
            evidence_source=evidence_source,
            confidence_score=confidence,
            retrieval_time_ms=retrieval_time_ms,
            citation_id=citation_id
        )

        self.state.retrieval_results.append(result)
        self.state.queries_answered += 1

        # Audit the retrieval
        self._audit_event(
            event_type="retrieval",
            details={
                "query_id": query.query_id,
                "evidence_retrieved": retrieved,
                "evidence_source": evidence_source,
                "confidence_score": confidence,
                "retrieval_time_ms": retrieval_time_ms
            }
        )

        logger.info(f"Retrieval complete: {query.query_id} | Evidence={retrieved} | Source={evidence_source}")
        return result

    def should_integrate_evidence(self, result: RetrievalResult) -> Tuple[bool, str]:
        """Determine whether retrieved evidence should be integrated into generation.

        Args:
            result: RetrievalResult from retrieval phase

        Returns:
            Tuple of (should_integrate: bool, reasoning: str)
        """
        if self.retrieval_mode == RetrievalMode.NATIVE_ONLY:
            return False, "NATIVE_ONLY mode: evidence audit-only, not integrated into generation"

        if not result.evidence_retrieved:
            return False, f"No evidence retrieved for {result.query_id}"

        if result.confidence_score < 0.7:
            return False, f"Confidence score {result.confidence_score} below threshold (0.7)"

        return True, f"Evidence from {result.evidence_source} will be integrated (confidence={result.confidence_score})"

    def record_integration(self, result: RetrievalResult, integrated: bool, context_position: Optional[str] = None) -> None:
        """Record evidence integration into the generation pipeline.

        Args:
            result: RetrievalResult that was integrated or audited
            integrated: Whether evidence was actually integrated into generation
            context_position: Position in context (e.g., "prefix", "suffix", "none")
        """
        should_integrate, reasoning = self.should_integrate_evidence(result)

        integration_record = {
            "query_id": result.query_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "result_integration_attempted": integrated,
            "result_evidence_available": result.evidence_retrieved,
            "result_should_integrate": should_integrate,
            "context_position": context_position or "none",
            "reasoning": reasoning
        }

        self.state.integration_points.append(integration_record)

        # Audit the integration decision
        self._audit_event(
            event_type="integration",
            details=integration_record
        )

        logger.info(f"Integration recorded for {result.query_id}: integrated={integrated}, should_integrate={should_integrate}")

    def _search_native_cache(self, query_text: str) -> Optional[str]:
        """Search native model context window cache (simulated).

        Args:
            query_text: Query to search for

        Returns:
            Evidence text if found, None otherwise
        """
        # In a real system, this would search the KV cache
        # For deterministic testing, return None (native mode doesn't have external context)
        return None

    def _search_external_kb(self, query_text: str) -> Tuple[Optional[str], bool]:
        """Search external knowledge base.

        Args:
            query_text: Query to search for

        Returns:
            Tuple of (evidence_text, found)
        """
        if not self.external_kb:
            return None, False

        # Simple keyword matching
        for doc_id, doc_text in self.external_kb.items():
            if any(word.lower() in doc_text.lower() for word in query_text.split()):
                return doc_text, True

        return None, False

    def register_external_knowledge_base(self, kb: Dict[str, str]) -> None:
        """Register or update external knowledge base for retrieval.

        Args:
            kb: Dictionary mapping doc IDs to document text
        """
        self.external_kb = kb
        logger.info(f"External KB registered with {len(kb)} documents")

    def _audit_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Record an audit event.

        Args:
            event_type: Type of event (query, retrieval, integration, audit)
            details: Event details
        """
        event_id = f"AUDIT-{self.experiment_id}-{len(self.state.audit_events):05d}"
        event = AuditEvent(
            event_id=event_id,
            experiment_id=self.experiment_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type=event_type,
            details=details
        )
        self.state.audit_events.append(event)

    def save_audit_log(self, path: Path = None) -> Path:
        """Save audit log to file.

        Args:
            path: Output file path; defaults to results/{experiment_id}_rag_audit.jsonl

        Returns:
            Path to saved audit log
        """
        if path is None:
            path = self.output_dir / f"{self.experiment_id}_rag_audit.jsonl"

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            for event in self.state.audit_events:
                f.write(json.dumps(asdict(event)) + '\n')

        logger.info(f"Audit log saved to {path}")
        return path

    def save_retrieval_results(self, path: Path = None) -> Path:
        """Save retrieval results to file.

        Args:
            path: Output file path; defaults to results/{experiment_id}_rag_results.jsonl

        Returns:
            Path to saved results
        """
        if path is None:
            path = self.output_dir / f"{self.experiment_id}_rag_results.jsonl"

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            for result in self.state.retrieval_results:
                result_dict = asdict(result)
                # Note: evidence_text may be large; include in audit
                f.write(json.dumps(result_dict) + '\n')

        logger.info(f"Retrieval results saved to {path}")
        return path

    def get_retrieval_audit_report(self) -> Dict[str, Any]:
        """Generate a comprehensive retrieval and audit report.

        Returns:
            Dictionary with retrieval metrics and audit summary
        """
        elapsed_ms = (time.time() - self.state.start_time) * 1000

        # Categorize retrieval results
        successful_retrievals = sum(1 for r in self.state.retrieval_results if r.evidence_retrieved)
        failed_retrievals = len(self.state.retrieval_results) - successful_retrievals

        # Categorize integration decisions
        integrated_count = sum(1 for ip in self.state.integration_points if ip.get("result_integration_attempted"))
        audit_only_count = len(self.state.integration_points) - integrated_count

        # Identify retrieval context sources
        sources = {}
        for result in self.state.retrieval_results:
            source = result.evidence_source
            sources[source] = sources.get(source, 0) + 1

        return {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "elapsed_ms": elapsed_ms,
            "retrieval_mode": self.retrieval_mode,
            "queries": {
                "issued": self.state.queries_issued,
                "answered": self.state.queries_answered
            },
            "retrieval": {
                "operations": self.state.retrieval_events,
                "successful": successful_retrievals,
                "failed": failed_retrievals,
                "success_rate": (
                    successful_retrievals / self.state.retrieval_events * 100
                    if self.state.retrieval_events > 0 else 0
                )
            },
            "evidence_sources": sources,
            "integration": {
                "points_recorded": len(self.state.integration_points),
                "evidence_integrated": integrated_count,
                "audit_only_queries": audit_only_count
            },
            "audit": {
                "total_events": len(self.state.audit_events),
                "audit_completeness": "complete" if len(self.state.audit_events) >= 3 * self.state.queries_issued else "partial"
            },
            "note": f"Retrieval mode: {self.retrieval_mode}. If NATIVE_ONLY, all retrieval is audit-only and not integrated."
        }
