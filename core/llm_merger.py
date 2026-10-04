"""LLM Merge Engine powered by Google GenAI (Gemini Flash).

Intelligently evaluates semantic overlap between candidate Kindle highlights and
existing Obsidian note excerpts to synthesize deltas or discard conceptual duplicates.
Protected by token-bucket rate limiting and jittered exponential backoff.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Callable, List, Optional
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from core.models import ActionType, KindleHighlight, LLMMergeResponse, VaultParagraph
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
MODEL_CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]


class _StructuredSchema(BaseModel):
    action: str = Field(
        description="One of: MERGE_APPEND, NEW_SECTION, STANDALONE_NOTE, DISCARD_DUPLICATE"
    )
    reasoning: str = Field(
        description="Concise rationale for why this merge/routing action was chosen"
    )
    target_section: Optional[str] = Field(
        default=None,
        description="Markdown section title (without '##') to place or append the content into",
    )
    content_to_insert: str = Field(
        description="Clean Markdown text ready to write into Obsidian, including source citation [[Book Title]]"
    )
    suggested_tags: List[str] = Field(
        default_factory=list,
        description="List of relevant lowercase tags (without '#') for Obsidian YAML frontmatter",
    )


class LLMMerger:
    """Manages structured LLM delta synthesis using Gemini Flash with automatic rate limiting."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        min_request_interval: Optional[float] = None,
        max_retries_per_model: int = 5,
        rate_limiter: Optional[RateLimiter] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.preferred_model = model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

        # Determine rate limit interval (default: 4.2 seconds for Free Tier safety)
        if min_request_interval is not None:
            self.min_request_interval = float(min_request_interval)
        else:
            env_interval = os.environ.get("GEMINI_REQUEST_INTERVAL")
            self.min_request_interval = (
                float(env_interval) if env_interval is not None else DEFAULT_MIN_INTERVAL_SECONDS
            )

        self.rate_limiter = rate_limiter or RateLimiter(min_interval_seconds=self.min_request_interval)
        self.max_retries_per_model = max_retries_per_model
        self.status_callback = status_callback
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        """Lazy loader for genai client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not set. Please provide it in .env or via the application UI."
                )
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _enforce_rate_limit(self) -> None:
        """Throttle requests to guarantee compliance with Free Tier quota."""
        self.rate_limiter.acquire(
            on_cooldown=lambda cd: self.status_callback and self.status_callback(
                f"Rate limit cooldown: waiting {cd:.1f}s before Gemini call..."
            )
        )

    def synthesize_merge(
        self,
        highlight: KindleHighlight,
        matched_paragraph: Optional[VaultParagraph],
        similarity_score: float,
    ) -> LLMMergeResponse:
        """Call Gemini Flash with structured JSON output to determine the merge strategy."""
        excerpt_text = matched_paragraph.text if matched_paragraph else "None (New note or empty section)"
        section_name = matched_paragraph.section_heading if matched_paragraph else "Highlights"

        prompt = f"""You are an expert knowledge curator and personal PKM (Personal Knowledge Management) assistant.
Your task is to analyze a new Kindle highlight against an existing note excerpt in an Obsidian vault and determine the optimal synthesis action.

### Context:
- Target Book: "{highlight.clean_title}" by {highlight.author}
- Highlight Location: {highlight.location or 'Unknown'}
- Cosine Similarity to existing excerpt: {similarity_score:.3f}

### Existing Note Excerpt (Section: '{section_name}'):
\"\"\"{excerpt_text}\"\"\"

### Candidate Kindle Highlight:
\"\"\"{highlight.text}\"\"\"

### Routing Actions:
1. "MERGE_APPEND": The highlight adds nuance, a concrete example, a supporting argument, or additional detail directly extending the existing excerpt. Synthesize or append it cleanly into the target section. Format math with Obsidian LaTeX ($...$, $$...$$).
2. "NEW_SECTION": The highlight relates to the broader book or theme, but introduces a distinct subtopic or separate idea requiring its own section heading.
3. "STANDALONE_NOTE": The highlight articulates a standalone universal principle, mental model, or major concept that merits its own atomic note linked back to [[{highlight.clean_title}]].
4. "DISCARD_DUPLICATE": The highlight expresses the exact same semantic thought, quote, or fact already captured in the existing note excerpt with zero incremental value.

### Output Formatting Instructions:
- "action": Exactly one of ["MERGE_APPEND", "NEW_SECTION", "STANDALONE_NOTE", "DISCARD_DUPLICATE"].
- "target_section": Section heading without '##' (e.g. "{section_name}" or a descriptive new heading).
- "content_to_insert": Polished markdown. Must include citation: quote formatting `> "{highlight.text}"` or synthesis bullet, citation link `[[{highlight.clean_title}]]`, and location. Format any formulas in LaTeX.
- "suggested_tags": 2 to 4 clean lowercase topical tags (e.g. ["mental-models", "decision-making"]).
"""

        models_to_try = [self.preferred_model] + [
            m for m in MODEL_CANDIDATES if m != self.preferred_model
        ]

        last_error = None
        for model_name in models_to_try:
            try:
                self._enforce_rate_limit()

                @retry_with_exponential_backoff(
                    max_retries=self.max_retries_per_model,
                    base_delay=2.0,
                    on_retry=lambda att, wait, err: self.status_callback and self.status_callback(
                        f"429 Quota reached on {model_name}. Backoff waiting {wait:.1f}s (retry {att}/{self.max_retries_per_model})..."
                    ),
                )
                def _execute_api_call():
                    return self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=_StructuredSchema,
                            temperature=0.2,
                            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                        ),
                    )

                response = _execute_api_call()

                if response and response.text:
                    data = json.loads(response.text)
                    raw_action = data.get("action", "MERGE_APPEND").upper()
                    if raw_action not in [a.value for a in ActionType]:
                        raw_action = ActionType.MERGE_APPEND.value

                    return LLMMergeResponse(
                        action=ActionType(raw_action),
                        reasoning=data.get("reasoning", "Synthesized by Gemini Flash"),
                        target_section=data.get("target_section") or section_name,
                        content_to_insert=data.get(
                            "content_to_insert",
                            f"> {highlight.text}\n>\n> *— [[{highlight.clean_title}]], {highlight.location}*",
                        ),
                        suggested_tags=data.get("suggested_tags", []),
                    )

            except Exception as e:
                logger.warning("Gemini model %s failed: %s", model_name, e)
                last_error = e
                if is_decommissioned_model_error(e):
                    continue
                continue

        # If LLM fails across all models, fallback gracefully to MERGE_APPEND
        logger.error("All Gemini model attempts failed. Falling back to local formatting: %s", last_error)
        fallback_content = (
            f"> {highlight.text}\n>\n"
            f"> *Source: [[{highlight.clean_title}]] (Location: {highlight.location})*"
        )
        return LLMMergeResponse(
            action=ActionType.MERGE_APPEND,
            reasoning=f"LLM synthesis fallback (API error: {str(last_error)[:100]})",
            target_section=section_name,
            content_to_insert=fallback_content,
            suggested_tags=["kindle-highlight"],
        )
