"""Kindle to Obsidian Sync Engine with Hybrid Vector Deduplication and Conceptual LaTeX Synthesis."""

from core.models import (
    ActionType,
    BookSynthesisResult,
    HighlightType,
    KindleHighlight,
    LLMMergeResponse,
    SyncItemAction,
    SyncSummary,
    ThemeNoteSynthesis,
    VaultParagraph,
)
from core.parser import clean_filename, group_highlights_by_book, parse_clippings_content, parse_clippings_file
from core.rate_limiter import RateLimiter, retry_with_exponential_backoff
from core.router import VectorRouter
from core.synthesizer import BookSynthesizer
from core.vault import VaultManager
from core.vector_index import VaultVectorIndex
from core.llm_merger import LLMMerger
from core.pipeline import SyncPipeline

__all__ = [
    "ActionType",
    "BookSynthesisResult",
    "BookSynthesizer",
    "HighlightType",
    "KindleHighlight",
    "LLMMergeResponse",
    "LLMMerger",
    "RateLimiter",
    "SyncItemAction",
    "SyncPipeline",
    "SyncSummary",
    "ThemeNoteSynthesis",
    "VaultManager",
    "VaultParagraph",
    "VaultVectorIndex",
    "VectorRouter",
    "clean_filename",
    "group_highlights_by_book",
    "parse_clippings_content",
    "parse_clippings_file",
    "retry_with_exponential_backoff",
]
