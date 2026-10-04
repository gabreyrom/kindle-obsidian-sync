"""Hybrid Vector Routing Engine.

Evaluates cosine similarity S_max between a candidate highlight and existing
vault paragraphs using a 3-tier gating mechanism:
  - S_max >= 0.86: DISCARD_DUPLICATE (bypasses LLM, zero cost)
  - S_max < 0.62: NEW_SECTION / STANDALONE_NOTE (bypasses LLM, orthogonal concept)
  - 0.62 <= S_max < 0.86: ROUTE_TO_LLM (semantic overlap boundary)
"""

from __future__ import annotations

from typing import Optional
from core.models import ActionType, RouterResult, VaultParagraph

DEFAULT_DUPLICATE_THRESHOLD = 0.86
DEFAULT_NOVEL_THRESHOLD = 0.62


class VectorRouter:
    """Evaluates candidate highlights against indexed note paragraphs and determines the routing action."""

    def __init__(
        self,
        duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
        novel_threshold: float = DEFAULT_NOVEL_THRESHOLD,
    ):
        self.duplicate_threshold = duplicate_threshold
        self.novel_threshold = novel_threshold

    def route(
        self,
        max_similarity: float,
        best_match: Optional[VaultParagraph] = None,
    ) -> RouterResult:
        """Apply the 3-tier routing gate given the maximum cosine similarity."""
        s = float(max_similarity)

        # Tier 1: Duplicate zone (high semantic and lexical congruence)
        if s >= self.duplicate_threshold:
            matched_snippet = best_match.text[:120] if best_match else ""
            reasoning = (
                f"S_max ({s:.3f}) >= {self.duplicate_threshold:.2f}: "
                f"Candidate highlight is redundant with existing note excerpt "
                f"('{matched_snippet}...'). Discarding duplicate without LLM invocation."
            )
            return RouterResult(
                action=ActionType.DISCARD_DUPLICATE,
                max_similarity=s,
                matched_paragraph=best_match,
                matched_text=best_match.text if best_match else None,
                reasoning=reasoning,
            )

        # Tier 2: Novel/Orthogonal zone (low semantic overlap)
        if s < self.novel_threshold:
            reasoning = (
                f"S_max ({s:.3f}) < {self.novel_threshold:.2f}: "
                f"Candidate highlight represents an orthogonal concept with no strong "
                f"conceptual anchor in existing notes. Bypassing LLM to create a new section."
            )
            return RouterResult(
                action=ActionType.NEW_SECTION,
                max_similarity=s,
                matched_paragraph=best_match,
                matched_text=best_match.text if best_match else None,
                reasoning=reasoning,
            )

        # Tier 3: Semantic Overlap Boundary Zone (Requires LLM synthesis)
        matched_snippet = best_match.text[:120] if best_match else ""
        reasoning = (
            f"{self.novel_threshold:.2f} <= S_max ({s:.3f}) < {self.duplicate_threshold:.2f}: "
            f"Semantic overlap boundary detected against '{matched_snippet}...'. "
            f"Routing to Gemini Flash for intelligent delta synthesis."
        )
        return RouterResult(
            action=ActionType.ROUTE_TO_LLM,
            max_similarity=s,
            matched_paragraph=best_match,
            matched_text=best_match.text if best_match else None,
            reasoning=reasoning,
        )
