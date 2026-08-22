"""
guardrails/input_guardrail.py

Input Guardrail for checking:
1. Safety & Off-topic detection (pre-retrieval)
2. Low retrieval confidence threshold (< 0.40)
"""

import re
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# List of dangerous/unsafe keywords or jailbreak injection patterns
UNSAFE_KEYWORDS = [
    r"\bbomb\b", r"\bweapon\b", r"\bexplosive\b", r"\bhack into\b",
    r"\bmalware\b", r"\bpoison\b", r"\bsuicide\b", r"\billegal\b",
    r"ignore previous instructions", r"system promptoverride"
]

OFF_TOPIC_KEYWORDS = [
    r"\bquantum quantum quantum\b", r"\bqwertyuiop\b", r"\basdfghjkl\b"
]


class InputGuardrail:
    """Pre-retrieval input guardrail evaluating safety and retrieval score confidence."""

    def __init__(self, confidence_threshold: float = 0.45):
        self.confidence_threshold = confidence_threshold

    def check_safety(self, query_text: str) -> Tuple[bool, str]:
        """Checks if query contains unsafe, malicious, or prompt injection patterns."""
        if not query_text or not query_text.strip():
            return False, "Empty query provided."

        text_lower = query_text.lower()
        for pattern in UNSAFE_KEYWORDS:
            if re.search(pattern, text_lower):
                logger.warning(f"Input Guardrail triggered unsafe pattern match: '{pattern}'")
                return False, f"Query violates safety policies (matched pattern: {pattern})."

        for pattern in OFF_TOPIC_KEYWORDS:
            if re.search(pattern, text_lower):
                return False, "Query contains meaningless off-topic text."

        return True, "Query passed safety check."

    def check_retrieval_confidence(self, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[bool, float, str]:
        """Checks if top retrieval similarity score meets the minimum confidence threshold (< 0.40)."""
        if not retrieved_chunks:
            logger.info("Retrieval returned 0 chunks. Flagging low confidence.")
            return False, 0.0, "No relevant context chunks found."

        top_score = retrieved_chunks[0].get("hybrid_score", retrieved_chunks[0].get("score", 0.0))
        
        if top_score < self.confidence_threshold:
            logger.info(f"Top retrieval similarity score ({top_score:.4f}) below threshold ({self.confidence_threshold}).")
            return False, top_score, f"Top retrieval confidence score ({top_score:.4f}) is below minimum threshold ({self.confidence_threshold})."

        return True, top_score, f"Retrieval confidence passed ({top_score:.4f} >= {self.confidence_threshold})."
