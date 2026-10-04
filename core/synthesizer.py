"""Conceptual Synthesis & LaTeX Equation Extraction Engine.

Ingests all parsed Kindle highlights for a book in a single pass to preserve global context,
clusters them into 3 to 6 atomic conceptual themes using Gemini Flash, reconstructs mathematical
models into Obsidian-compatible LaTeX, and outputs structured theme notes.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Callable, List, Optional
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from core.models import BookSynthesisResult, KindleHighlight, ThemeNoteSynthesis
from core.rate_limiter import (
    DEFAULT_MIN_INTERVAL_SECONDS,
    RateLimiter,
    extract_retry_delay,
    is_decommissioned_model_error,
    is_rate_limit_error,
    retry_with_exponential_backoff,
)

logger = logging.getLogger(__name__)

# Fallback cascade of Gemini Flash models for high reliability
SYNTHESIZER_MODEL_CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]


class _ThemeSchema(BaseModel):
    theme_name: str = Field(
        description="Clean, concise theme title suitable for a filename (e.g. 'Expected Goals & Shot Quality Models')"
    )
    core_concept: str = Field(
        description="1-2 sentence rigorous theoretical summary explaining the conceptual foundation"
    )
    formal_model_and_equations: str = Field(
        description=(
            "Obsidian-compatible LaTeX mathematical models, equations ($...$, $$...$$), and variable definitions. "
            "Define all symbols and parameters clearly. If non-mathematical, formalize the analytical framework with symbolic logic."
        )
    )
    analytical_takeaways: List[str] = Field(
        description="3 to 5 analytical takeaways aggregating evidence and insights across highlights"
    )
    source_evidence: List[str] = Field(
        description="Direct quotes from the input highlights formatted as blockquotes with Kindle citations"
    )
    suggested_tags: List[str] = Field(
        description="2 to 4 topical lowercase tags without '#'"
    )


class _BookSynthesisSchema(BaseModel):
    synopsis: str = Field(
        description="High-level 3-4 sentence theoretical and practical overview of the book's overarching framework"
    )
    overall_tags: List[str] = Field(
        description="3 to 5 topical lowercase tags without '#'"
    )
    themes: List[_ThemeSchema] = Field(
        description="List of 3 to 6 atomic conceptual themes clustering all provided highlights"
    )


class BookSynthesizer:
    """Manages full-book thematic clustering and LaTeX mathematical reconstruction."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        min_request_interval: float = DEFAULT_MIN_INTERVAL_SECONDS,
        rate_limiter: Optional[RateLimiter] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.preferred_model = model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        self.rate_limiter = rate_limiter or RateLimiter(min_interval_seconds=min_request_interval)
        self.status_callback = status_callback
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        """Lazy loader for genai client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is not set. Please provide it in .env or via the application UI.")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def synthesize_book(
        self,
        book_title: str,
        author: str,
        highlights: List[KindleHighlight],
    ) -> BookSynthesisResult:
        """Synthesize all highlights for a book into 3-6 conceptual theme notes with LaTeX math."""
        if not highlights:
            return BookSynthesisResult(
                book_title=book_title,
                author=author,
                synopsis=f"No highlights available for {book_title}.",
                overall_tags=["book"],
                themes=[],
            )

        # Build clean highlight block
        highlight_lines: List[str] = []
        for i, h in enumerate(highlights, 1):
            loc_str = f"Location {h.location}" if h.location else "Kindle"
            highlight_lines.append(f"Highlight #{i} ({loc_str}):\n\"{h.text.strip()}\"")
        highlights_block = "\n\n".join(highlight_lines)

        prompt = f"""You are an elite research scientist, quantitative analyst, and personal knowledge management expert.
Your mission is to perform deep conceptual synthesis on a collection of reading highlights from the book "{book_title}" by {author}.

### Mathematical Reconstruction & Formalization Objective:
1. Identify all core theoretical concepts, quantitative models, statistical estimators, metrics, or analytical frameworks referenced throughout the highlights (e.g. regression models, conditional expectations $\\mathbb{{E}}[Y|X]$, causal inference / diff-in-diff, Bayesian priors/posteriors, shot quality / expected goals $xG$, loss functions, optimization bounds, or formal decision rules).
2. For EVERY theme, reconstruct and format mathematical expressions into clean, professional Obsidian-compatible LaTeX:
   - Use $...$ for inline math notation (e.g. $P(Y=1|X)$, $\\beta_1$, $\\mathbb{{E}}[xG]$).
   - Use $$...$$ for standalone display equations.
   - Explicitly define variables, parameters, and estimators.
   - If a theme is conceptual/non-mathematical, formalize its analytical relationship, logical proof, or structural decision tree with formal mathematical/logical notation.
3. Cluster the highlights into exactly 3 to 6 atomic, distinct conceptual themes. Do NOT create duplicate or superficial themes.
4. For each theme:
   - "theme_name": Concise, descriptive title suitable for a filename (no illegal characters like / \\ : * ? " < > |).
   - "core_concept": 1-2 sentence theoretical summary.
   - "formal_model_and_equations": Explicit LaTeX math expressions and parameter definitions.
   - "analytical_takeaways": 3-5 synthesized analytical takeaways aggregating evidence.
   - "source_evidence": Verbatim blockquotes from the highlights with Kindle citations.
   - "suggested_tags": 2-4 lowercase tags.

### Input Highlights ({len(highlights)} items):
{highlights_block}
"""

        models_to_try = [self.preferred_model] + [
            m for m in SYNTHESIZER_MODEL_CANDIDATES if m != self.preferred_model
        ]

        last_error = None
        for model_name in models_to_try:
            try:
                # Enforce rate-limiting interval before API call
                if self.status_callback:
                    self.status_callback(f"Pacing API call for {model_name}...")
                self.rate_limiter.acquire(
                    on_cooldown=lambda cd: self.status_callback and self.status_callback(
                        f"Rate limit cooldown: waiting {cd:.1f}s before calling {model_name}..."
                    )
                )

                # Execute with exponential backoff decorator
                @retry_with_exponential_backoff(
                    max_retries=5,
                    base_delay=2.0,
                    on_retry=lambda att, wait, err: self.status_callback and self.status_callback(
                        f"429 Quota reached on {model_name}. Backoff waiting {wait:.1f}s (retry {att}/5)..."
                    ),
                )
                def _call_gemini():
                    return self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=_BookSynthesisSchema,
                            temperature=0.2,
                            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                        ),
                    )

                response = _call_gemini()

                if response and response.text:
                    data = json.loads(response.text)
                    raw_themes = data.get("themes", [])
                    themes: List[ThemeNoteSynthesis] = []
                    for t in raw_themes:
                        themes.append(
                            ThemeNoteSynthesis(
                                theme_name=t.get("theme_name", "Core Concept"),
                                core_concept=t.get("core_concept", "Theoretical synthesis."),
                                formal_model_and_equations=t.get(
                                    "formal_model_and_equations", "$$\\text{Conceptual framework pending formalization}$$"
                                ),
                                analytical_takeaways=t.get("analytical_takeaways", []),
                                source_evidence=t.get("source_evidence", []),
                                suggested_tags=t.get("suggested_tags", ["concept"]),
                            )
                        )

                    return BookSynthesisResult(
                        book_title=book_title,
                        author=author,
                        synopsis=data.get("synopsis", f"Thematic knowledge synthesis for {book_title}."),
                        overall_tags=data.get("overall_tags", ["book-overview"]),
                        themes=themes,
                    )

            except Exception as e:
                logger.warning("Synthesis model %s failed: %s", model_name, e)
                last_error = e
                if is_decommissioned_model_error(e):
                    continue
                continue

        logger.error("All Gemini model synthesis attempts failed: %s", last_error)
        # Graceful fallback: produce single theme note with raw highlights
        fallback_theme = ThemeNoteSynthesis(
            theme_name="Key Highlights & Notes",
            core_concept=f"Consolidated highlights from {book_title} (offline fallback).",
            formal_model_and_equations="*Mathematical reconstruction unavailable due to API rate limit.*",
            analytical_takeaways=[
                f"Parsed {len(highlights)} highlights across the volume.",
                "Review individual source citations below.",
            ],
            source_evidence=[
                f"> \"{h.text}\"\n>\n> *— Kindle Location {h.location or 'N/A'}*"
                for h in highlights[:20]
            ],
            suggested_tags=["highlights", "reading-notes"],
        )

        return BookSynthesisResult(
            book_title=book_title,
            author=author,
            synopsis=f"Autonomous highlight compilation for {book_title} (synthesized via local fallback).",
            overall_tags=["book-overview", "fallback-sync"],
            themes=[fallback_theme],
        )
