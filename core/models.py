"""Data models for Kindle Obsidian Sync."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class HighlightType(str, Enum):
    HIGHLIGHT = "Highlight"
    NOTE = "Note"
    BOOKMARK = "Bookmark"


class ActionType(str, Enum):
    MERGE_APPEND = "MERGE_APPEND"
    NEW_SECTION = "NEW_SECTION"
    STANDALONE_NOTE = "STANDALONE_NOTE"
    DISCARD_DUPLICATE = "DISCARD_DUPLICATE"
    ROUTE_TO_LLM = "ROUTE_TO_LLM"


class KindleHighlight(BaseModel):
    """Represents a single parsed clipping from a Kindle."""
    book_title: str
    clean_title: str
    author: str = "Unknown"
    location: str = ""
    page: Optional[str] = None
    timestamp: Optional[str] = None
    text: str
    highlight_type: HighlightType = HighlightType.HIGHLIGHT
    clipping_id: Optional[str] = None


class VaultParagraph(BaseModel):
    """Represents an extracted paragraph/block from an existing Obsidian note."""
    file_path: str
    rel_path: str
    note_title: str
    section_heading: Optional[str] = None
    paragraph_index: int = 0
    text: str


class RouterResult(BaseModel):
    """Output from the cosine similarity vector gating."""
    action: ActionType
    max_similarity: float
    matched_paragraph: Optional[VaultParagraph] = None
    matched_text: Optional[str] = None
    reasoning: str


class LLMMergeResponse(BaseModel):
    """Structured response schema enforced from Gemini."""
    action: ActionType = Field(
        description="The action to take: MERGE_APPEND, NEW_SECTION, STANDALONE_NOTE, or DISCARD_DUPLICATE"
    )
    reasoning: str = Field(
        description="Explanation for why this action was selected"
    )
    target_section: Optional[str] = Field(
        default=None,
        description="Markdown section heading (without ##) to append to or create"
    )
    content_to_insert: str = Field(
        description="Markdown formatted text with source citation [[Book Title]]"
    )
    suggested_tags: List[str] = Field(
        default_factory=list,
        description="Extracted topical tags for Obsidian YAML frontmatter"
    )


class SyncItemAction(BaseModel):
    """Record of execution outcome for a single highlight."""
    highlight: KindleHighlight
    router_action: ActionType
    similarity_score: float
    final_action: ActionType
    reasoning: str
    target_file: str
    target_section: Optional[str] = None
    content_written: Optional[str] = None
    suggested_tags: List[str] = Field(default_factory=list)
    llm_invoked: bool = False


class ThemeNoteSynthesis(BaseModel):
    """Structured representation of a single atomic conceptual theme note."""
    theme_name: str = Field(
        description="Clean, concise theme title suitable for an Obsidian filename (e.g., 'Expected Goals & Shot Quality Models')"
    )
    core_concept: str = Field(
        description="1-2 sentence theoretical summary explaining the conceptual foundation"
    )
    formal_model_and_equations: str = Field(
        description="LaTeX mathematical models, equations ($...$, $$...$$), and variable definitions"
    )
    analytical_takeaways: List[str] = Field(
        default_factory=list,
        description="Synthesized bullet points aggregating analytical evidence across highlights"
    )
    source_evidence: List[str] = Field(
        default_factory=list,
        description="Blockquotes with Kindle location citations supporting this theme"
    )
    suggested_tags: List[str] = Field(
        default_factory=list,
        description="Topical lowercase tags without '#'"
    )


class BookSynthesisResult(BaseModel):
    """Complete multi-note thematic synthesis for a book."""
    book_title: str
    author: str = "Unknown"
    synopsis: str = Field(
        description="High-level theoretical and practical overview of the book's core theses"
    )
    overall_tags: List[str] = Field(
        default_factory=list,
        description="Top-level topical tags for the 00 - Overview note"
    )
    themes: List[ThemeNoteSynthesis] = Field(
        default_factory=list,
        description="List of 3 to 6 atomic conceptual themes"
    )


class SyncSummary(BaseModel):
    """Telemetry and summary metrics of a sync batch."""
    total_processed: int = 0
    duplicates_discarded: int = 0
    direct_inserts: int = 0
    llm_calls_made: int = 0
    merged_appends: int = 0
    new_sections: int = 0
    standalone_notes: int = 0
    themes_generated: int = 0
    notes_created: List[str] = Field(default_factory=list)
    items: List[SyncItemAction] = Field(default_factory=list)

    @property
    def llm_calls_saved_percent(self) -> float:
        if self.total_processed == 0:
            return 100.0
        saved = self.total_processed - self.llm_calls_made
        return max(0.0, min(100.0, (saved / self.total_processed) * 100.0))

