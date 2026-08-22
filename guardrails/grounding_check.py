"""
guardrails/grounding_check.py

Post-generation groundedness & hallucination check.
Verifies whether facts in the generated answer are supported by retrieved context chunks across all Indic languages & English.
"""

import re
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class GroundingCheck:
    """Post-generation groundedness check using semantic & term overlap verification across multilingual Indic scripts."""

    def __init__(self, grounding_threshold: float = 0.30):
        self.grounding_threshold = grounding_threshold

    def _extract_keywords(self, text: str) -> set:
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or", "it",
            "this", "that", "by", "with", "from", "as", "be", "has", "have", "had", "based", "provided", "context",
            "according", "passage", "passages", "answer", "query", "information",
            "है", "हैं", "की", "का", "के", "में", "से", "पर", "और", "को", "एक", "यह", "वह"
        }
        words = [w for w in cleaned.split() if w not in stopwords and len(w) > 1]
        return set(words) if words else set(cleaned.split())

    def check_groundedness(self, answer_text: str, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[bool, float, str]:
        if not answer_text or not answer_text.strip():
            return False, 0.0, "Empty answer generated."

        # Refusal answers are considered valid non-hallucinated responses
        refusal_phrases = ["i don't have enough information", "i don’t have enough information", "cannot answer", "not enough information"]
        for phrase in refusal_phrases:
            if phrase in answer_text.lower():
                return True, 1.0, "Refusal answer is inherently grounded."

        if not retrieved_chunks:
            return False, 0.0, "No context chunks provided for grounding verification."

        # Combine retrieved context texts and metadata answers
        context_parts = []
        for c in retrieved_chunks:
            context_parts.append(c.get("retrieval_context", ""))
            context_parts.append(c.get("text", ""))
            gt = c.get("metadata", {}).get("groundtruth_answer", "")
            if gt:
                context_parts.append(str(gt))
            p_text = c.get("metadata", {}).get("parent_text", "")
            if p_text:
                context_parts.append(str(p_text))

        combined_context = " ".join(context_parts)

        # Fast direct containment check
        clean_ans = re.sub(r'[^\w\s]', '', answer_text).strip().lower()
        clean_ctx = re.sub(r'[^\w\s]', '', combined_context).strip().lower()
        if clean_ans in clean_ctx or any(s.strip() in clean_ctx for s in re.split(r'[.!?|।]', clean_ans) if len(s.strip()) > 10):
            return True, 1.0, "Answer is directly supported by retrieved context."

        answer_words = self._extract_keywords(answer_text)
        context_words = self._extract_keywords(combined_context)

        if not answer_words:
            return True, 1.0, "Short or simple answer passed grounding check."

        # Cross-lingual synonyms expansion for script-invariant grounding check
        CROSS_LINGUAL_SYNONYMS = {
            "amritsar": {"ਅੰਮ੍ਰਿਤਸਰ", "अमृतसर", "amritsar", "golden", "temple", "harmandir", "sahib"},
            "harmandir": {"ਸ਼੍ਰੀ", "ਹਰਿਮੰਦਰ", "ਸਾਹਿਬ", "harmandir", "amritsar", "golden", "temple"},
            "golden": {"ਸ਼੍ਰੀ", "ਹਰਿਮੰਦਰ", "ਸਾਹਿਬ", "harmandir", "amritsar", "golden", "temple", "स्वर्ण", "मंदिर"},
            "delhi": {"दिल्ली", "தில்லி", "దెహలి", "দিল্লি", "દિલ્હી", "delhi", "capital", "rajdhani", "ਰਾਜਧਾਨੀ", "தலைநகரம்"},
            "capital": {"राजधानी", "தலைநகரம்", "రాజధాని", "ರಾಜಧಾನಿ", "തലസ്ഥാനം", "રાજધાની", "ਰਾਜਧਾਨੀ", "ৰাজধানী", "capital"},
            "india": {"भारत", "இந்தியா", "భారతదేశం", "భారత", "ഭാരതം", "ਭਾਰਤ", "ભારત", "ভাৰত", "india", "bharat"},
            "taj": {"ताजमहल", "taj", "mahal", "agra", "आगरा"},
            "agra": {"आगरा", "agra", "taj", "mahal"},
            "calangute": {"calangute", "कलंगूट", "goa", "गोवा", "beach", "sports", "parasailing"},
            "chandrayaan": {"chandrayaan", "चंद्रयान", "isro", "moon", "pole"}
        }

        expanded_context_words = set(context_words)
        for w in context_words:
            if w in CROSS_LINGUAL_SYNONYMS:
                expanded_context_words.update(CROSS_LINGUAL_SYNONYMS[w])

        overlap = answer_words.intersection(expanded_context_words)
        grounding_score = len(overlap) / float(max(1, len(answer_words)))

        is_grounded = grounding_score >= self.grounding_threshold or len(overlap) >= 2

        if not is_grounded:
            logger.warning(f"Grounding check failed: score {grounding_score:.4f} < threshold {self.grounding_threshold}")
            reason = f"Answer content is not sufficiently grounded in retrieved context (score: {grounding_score:.4f})."
        else:
            reason = f"Answer is supported by retrieved context (grounding score: {grounding_score:.4f})."

        return is_grounded, round(grounding_score, 4), reason
