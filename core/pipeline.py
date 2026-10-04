"""Orchestration pipeline for Kindle-to-Obsidian sync with hybrid vector deduplication and LaTeX synthesis."""

from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional

from core.llm_merger import LLMMerger
from core.models import (
    ActionType,
    KindleHighlight,
    SyncItemAction,
    SyncSummary,
    VaultParagraph,
)
from core.router import VectorRouter
from core.synthesizer import BookSynthesizer
from core.vault import VaultManager
from core.vector_index import VaultVectorIndex

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, SyncItemAction], None]


class SyncPipeline:
    """Coordinates parsing, scoped vector retrieval, similarity gating, LLM merging, and vault persistence."""

    def __init__(
        self,
        vault_manager: VaultManager,
        vector_index: VaultVectorIndex,
        router: VectorRouter,
        llm_merger: LLMMerger,
        synthesizer: Optional[BookSynthesizer] = None,
    ):
        self.vault_manager = vault_manager
        self.vector_index = vector_index
        self.router = router
        self.llm_merger = llm_merger
        self.synthesizer = synthesizer

    def sync_highlights(
        self,
        highlights: List[KindleHighlight],
        progress_callback: Optional[ProgressCallback] = None,
        synthesize_concept_notes: bool = True,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> SyncSummary:
        """Process a collection of highlights through the hybrid vector deduplication pipeline."""
        summary = SyncSummary(total_processed=len(highlights))
        if not highlights:
            return summary

        # Group highlights by target book
        book_groups: dict[str, List[KindleHighlight]] = {}
        for h in highlights:
            book_groups.setdefault(h.clean_title, []).append(h)

        total_highlights_count = len(highlights)
        processed_counter = 0

        for clean_title, book_highlights in book_groups.items():
            author = book_highlights[0].author if book_highlights else "Unknown"
            has_existing = self.vault_manager.has_existing_notes(clean_title)

            # -------------------------------------------------------------
            # CASE 1: First-time sync with Conceptual Synthesis & LaTeX
            # If the book folder does not exist or has no markdown files,
            # skip vector deduplication (S_max = 0) and generate concept notes.
            # -------------------------------------------------------------
            if synthesize_concept_notes and not has_existing and self.synthesizer:
                if status_callback:
                    status_callback(f"Synthesizing LaTeX concept themes for '{clean_title}'...")

                summary.llm_calls_made += 1
                synthesis_result = self.synthesizer.synthesize_book(
                    book_title=clean_title,
                    author=author,
                    highlights=book_highlights,
                )

                # 1. Write 00 - Overview.md
                overview_path = self.vault_manager.write_overview_note(
                    clean_title=clean_title,
                    author=author,
                    synopsis=synthesis_result.synopsis,
                    themes=synthesis_result.themes,
                    overall_tags=synthesis_result.overall_tags,
                )
                summary.notes_created.append(overview_path)

                # 2. Write {Theme_Name}.md notes
                theme_paths: List[str] = []
                for theme in synthesis_result.themes:
                    t_path = self.vault_manager.write_theme_note(
                        clean_title=clean_title,
                        author=author,
                        theme=theme,
                    )
                    theme_paths.append(t_path)
                    summary.notes_created.append(t_path)

                summary.themes_generated += len(synthesis_result.themes)
                summary.new_sections += len(synthesis_result.themes)

                # Dynamically index generated notes for future queries
                scoped_paras = self.vault_manager.get_book_paragraphs(clean_title)
                if scoped_paras:
                    self.vector_index.index_paragraphs(scoped_paras)

                # Record per-highlight telemetry
                for h in book_highlights:
                    processed_counter += 1
                    assigned_theme = (
                        synthesis_result.themes[0].theme_name
                        if synthesis_result.themes
                        else "Overview"
                    )
                    target_file = theme_paths[0] if theme_paths else overview_path

                    item_action = SyncItemAction(
                        highlight=h,
                        router_action=ActionType.ROUTE_TO_LLM,
                        similarity_score=0.0,
                        final_action=ActionType.STANDALONE_NOTE,
                        reasoning=f"Synthesized into thematic note: '{assigned_theme}' with LaTeX math.",
                        target_file=target_file,
                        target_section="Formal Model & Equations",
                        content_written=f"Integrated into theme '{assigned_theme}'",
                        suggested_tags=synthesis_result.overall_tags,
                        llm_invoked=True,
                    )
                    summary.items.append(item_action)
                    if progress_callback:
                        progress_callback(processed_counter, total_highlights_count, item_action)

                continue

            # -------------------------------------------------------------
            # CASE 2: Incremental sync or Raw Highlights mode
            # Scoped Indexing Constraint: only index .md files inside this book folder!
            # -------------------------------------------------------------
            scoped_paragraphs = self.vault_manager.get_book_paragraphs(clean_title)
            if scoped_paragraphs:
                logger.info("Scoped indexing %d paragraphs for book '%s'...", len(scoped_paragraphs), clean_title)
                self.vector_index.index_paragraphs(scoped_paragraphs)
            else:
                self.vector_index.reset_collection()

            for highlight in book_highlights:
                processed_counter += 1

                max_sim, best_match, _ = self.vector_index.query_most_similar(highlight.text)
                router_result = self.router.route(max_sim, best_match)
                item_action: SyncItemAction

                # --- BRANCH A: Exact / Semantic Duplicate (S_max >= 0.86) ---
                if router_result.action == ActionType.DISCARD_DUPLICATE:
                    summary.duplicates_discarded += 1
                    book_path = self.vault_manager.get_book_note_path(highlight.clean_title)
                    item_action = SyncItemAction(
                        highlight=highlight,
                        router_action=router_result.action,
                        similarity_score=max_sim,
                        final_action=ActionType.DISCARD_DUPLICATE,
                        reasoning=router_result.reasoning,
                        target_file=book_path,
                        target_section=None,
                        content_written=None,
                        suggested_tags=[],
                        llm_invoked=False,
                    )

                # --- BRANCH B: Orthogonal / Novel Concept (S_max < 0.62) ---
                elif router_result.action == ActionType.NEW_SECTION:
                    summary.direct_inserts += 1
                    summary.new_sections += 1

                    loc_str = f"Location {highlight.location}" if highlight.location else "Kindle"
                    content_to_insert = (
                        f"> \"{highlight.text}\"\n>\n"
                        f"> *— [[00 - Overview|{highlight.clean_title}]], {loc_str}*"
                    )

                    target_section = "Key Concepts"
                    target_file, action_desc = self.vault_manager.apply_sync_action(
                        highlight=highlight,
                        action=ActionType.NEW_SECTION,
                        content_to_insert=content_to_insert,
                        target_section=target_section,
                        suggested_tags=["kindle-highlight"],
                    )

                    new_para = VaultParagraph(
                        file_path=target_file,
                        rel_path=os.path.relpath(target_file, self.vault_manager.vault_path),
                        note_title=highlight.clean_title,
                        section_heading=target_section,
                        paragraph_index=999,
                        text=highlight.text,
                    )
                    self.vector_index.add_single_paragraph(new_para)

                    item_action = SyncItemAction(
                        highlight=highlight,
                        router_action=router_result.action,
                        similarity_score=max_sim,
                        final_action=ActionType.NEW_SECTION,
                        reasoning=router_result.reasoning,
                        target_file=target_file,
                        target_section=target_section,
                        content_written=content_to_insert,
                        suggested_tags=["kindle-highlight"],
                        llm_invoked=False,
                    )

                # --- BRANCH C: Semantic Overlap Boundary (0.62 <= S_max < 0.86) -> LLM ---
                else:
                    summary.llm_calls_made += 1
                    llm_response = self.llm_merger.synthesize_merge(
                        highlight=highlight,
                        matched_paragraph=best_match,
                        similarity_score=max_sim,
                    )

                    if llm_response.action == ActionType.DISCARD_DUPLICATE:
                        summary.duplicates_discarded += 1
                        book_path = self.vault_manager.get_book_note_path(highlight.clean_title)
                        item_action = SyncItemAction(
                            highlight=highlight,
                            router_action=router_result.action,
                            similarity_score=max_sim,
                            final_action=ActionType.DISCARD_DUPLICATE,
                            reasoning=f"LLM Gated: {llm_response.reasoning}",
                            target_file=book_path,
                            target_section=None,
                            content_written=None,
                            suggested_tags=llm_response.suggested_tags,
                            llm_invoked=True,
                        )
                    else:
                        if llm_response.action == ActionType.MERGE_APPEND:
                            summary.merged_appends += 1
                        elif llm_response.action == ActionType.NEW_SECTION:
                            summary.new_sections += 1
                        elif llm_response.action == ActionType.STANDALONE_NOTE:
                            summary.standalone_notes += 1

                        target_file, _ = self.vault_manager.apply_sync_action(
                            highlight=highlight,
                            action=llm_response.action,
                            content_to_insert=llm_response.content_to_insert,
                            target_section=llm_response.target_section,
                            suggested_tags=llm_response.suggested_tags,
                        )

                        new_para = VaultParagraph(
                            file_path=target_file,
                            rel_path=os.path.relpath(target_file, self.vault_manager.vault_path),
                            note_title=highlight.clean_title,
                            section_heading=llm_response.target_section,
                            paragraph_index=999,
                            text=highlight.text,
                        )
                        self.vector_index.add_single_paragraph(new_para)

                        item_action = SyncItemAction(
                            highlight=highlight,
                            router_action=router_result.action,
                            similarity_score=max_sim,
                            final_action=llm_response.action,
                            reasoning=f"LLM Synthesized: {llm_response.reasoning}",
                            target_file=target_file,
                            target_section=llm_response.target_section,
                            content_written=llm_response.content_to_insert,
                            suggested_tags=llm_response.suggested_tags,
                            llm_invoked=True,
                        )

                summary.items.append(item_action)
                if progress_callback:
                    progress_callback(processed_counter, total_highlights_count, item_action)

        return summary
