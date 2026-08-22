"""
guardrails/output_guardrail.py

Output Guardrail for intercepting ungrounded or low-confidence responses
and formatting standardized refusal logic.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

DEFAULT_REFUSAL_MESSAGE = "I don't have enough information to answer that."


class OutputGuardrail:
    """Output guardrail enforcing refusal logic when retrieval confidence is low or answer is ungrounded."""

    def __init__(self, default_refusal: str = DEFAULT_REFUSAL_MESSAGE):
        self.default_refusal = default_refusal

    def process_output(
        self,
        raw_answer: str,
        safety_passed: bool,
        confidence_passed: bool,
        grounding_passed: bool,
        refusal_reason: str = ""
    ) -> Dict[str, Any]:
        
        # If any guardrail failed, override output with refusal message
        if not safety_passed or not confidence_passed or not grounding_passed:
            logger.info(f"Output Guardrail intercepted output. Reason: {refusal_reason}")
            return {
                "final_answer": self.default_refusal,
                "is_refused": True,
                "refusal_reason": refusal_reason,
                "original_answer": raw_answer
            }

        return {
            "final_answer": raw_answer,
            "is_refused": False,
            "refusal_reason": "N/A",
            "original_answer": raw_answer
        }
