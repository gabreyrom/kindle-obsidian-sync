"""End-to-end integration tests for VectorIndex, Router, and SyncPipeline."""

import os
import shutil
import tempfile
from unittest.mock import MagicMock
import pytest

from core.llm_merger import LLMMerger
from core.models import ActionType, KindleHighlight, LLMMergeResponse, VaultParagraph
from core.pipeline import SyncPipeline
from core.router import VectorRouter
from core.vault import VaultManager
from core.vector_index import VaultVectorIndex


def test_vector_index_and_router():
    index = VaultVectorIndex(collection_name="test_pipeline_idx")
    index.reset_collection()

    p1 = VaultParagraph(
        file_path="/tmp/Thinking, Fast and Slow.md",
        rel_path="Thinking, Fast and Slow.md",
        note_title="Thinking, Fast and Slow",
        section_heading="Cognitive Biases",
        paragraph_index=0,
        text="A reliable way to make people believe in falsehoods is frequent repetition, because familiarity is not easily distinguished from truth.",
    )
    index.index_paragraphs([p1])

    router = VectorRouter()

    # Query with exact quote
    query_exact = "A reliable way to make people believe in falsehoods is frequent repetition, because familiarity is not easily distinguished from truth."
    max_sim_exact, match_exact, _ = index.query_most_similar(query_exact)
    decision_exact = router.route(max_sim_exact, match_exact)
    assert max_sim_exact >= 0.95
    assert decision_exact.action == ActionType.DISCARD_DUPLICATE

    # Query with completely orthogonal topic
    query_orthogonal = "The mitochondria is the powerhouse of the cell in eukaryotic organisms."
    max_sim_orth, match_orth, _ = index.query_most_similar(query_orthogonal)
    decision_orth = router.route(max_sim_orth, match_orth)
    assert max_sim_orth < 0.62
    assert decision_orth.action == ActionType.NEW_SECTION


def test_sync_pipeline_with_mock_llm():
    temp_dir = tempfile.mkdtemp()
    try:
        vault = VaultManager(vault_path=temp_dir, default_folder="Books")
        index = VaultVectorIndex(collection_name="test_mock_pipeline")
        router = VectorRouter(duplicate_threshold=0.86, novel_threshold=0.62)

        # Mock LLM Merger
        mock_llm = MagicMock(spec=LLMMerger)
        mock_llm.synthesize_merge.return_value = LLMMergeResponse(
            action=ActionType.MERGE_APPEND,
            reasoning="Adds supporting evidence to cognitive biases",
            target_section="Cognitive Biases",
            content_to_insert="> System 1 uses cognitive ease to assess truth.\n>\n> *— [[Thinking, Fast and Slow]], Page 14*",
            suggested_tags=["psychology", "cognitive-bias"],
        )

        pipeline = SyncPipeline(vault, index, router, mock_llm)

        highlights = [
            # 1: Novel highlight (no existing note yet) -> should be NEW_SECTION
            KindleHighlight(
                book_title="Thinking, Fast and Slow",
                clean_title="Thinking, Fast and Slow",
                author="Daniel Kahneman",
                location="12",
                text="A reliable way to make people believe in falsehoods is frequent repetition.",
            ),
            # 2: Exact duplicate of 1 -> should be DISCARD_DUPLICATE
            KindleHighlight(
                book_title="Thinking, Fast and Slow",
                clean_title="Thinking, Fast and Slow",
                author="Daniel Kahneman",
                location="12",
                text="A reliable way to make people believe in falsehoods is frequent repetition.",
            ),
        ]

        summary = pipeline.sync_highlights(highlights)

        assert summary.total_processed == 2
        assert summary.new_sections == 1
        assert summary.duplicates_discarded == 1
        assert summary.direct_inserts == 1
        assert summary.llm_calls_made == 0  # LLM bypassed!
        assert summary.llm_calls_saved_percent == 100.0

        # Verify file exists
        note_path = vault.get_book_note_path("Thinking, Fast and Slow")
        assert os.path.exists(note_path)

        with open(note_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "A reliable way to make people believe in falsehoods" in content

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_pipeline_thematic_synthesis_with_mock_synthesizer():
    from core.synthesizer import BookSynthesizer
    from core.models import BookSynthesisResult, ThemeNoteSynthesis

    temp_dir = tempfile.mkdtemp()
    try:
        vault = VaultManager(vault_path=temp_dir, default_folder="Books")
        index = VaultVectorIndex(collection_name="test_thematic_pipeline")
        router = VectorRouter()
        mock_llm = MagicMock(spec=LLMMerger)

        mock_synth = MagicMock(spec=BookSynthesizer)
        mock_synth.synthesize_book.return_value = BookSynthesisResult(
            book_title="Football Hackers",
            author="Christoph Biermann",
            synopsis="The revolution in advanced football analytics, expected goals models, and spatial tracking.",
            overall_tags=["sports-analytics", "expected-goals", "machine-learning"],
            themes=[
                ThemeNoteSynthesis(
                    theme_name="Expected Goals & Shot Quality Models",
                    core_concept="Expected Goals (xG) measures the probability of a shot resulting in a goal based on spatial and contextual features.",
                    formal_model_and_equations="""$$\\text{xG}_i = \\sigma(\\beta_0 + \\beta_1 d_i + \\beta_2 \\theta_i + \\mathbf{x}_i^T \\boldsymbol{\\gamma})$$
Where $d_i$ is distance to goal, $\\theta_i$ is shot angle, and $\\sigma(z) = \\frac{1}{1 + e^{-z}}$ is the logistic sigmoid.""",
                    analytical_takeaways=[
                        "xG eliminates outcome bias by separating process quality from finishing variance.",
                        "Post-shot expected goals (xGOT) measures shot placement trajectory conditional on being on target.",
                    ],
                    source_evidence=[
                        "> \"Goals are rare events; assessing shot quality gives a much clearer signal of performance.\"\n>\n> *— Kindle Location 120*"
                    ],
                    suggested_tags=["xg", "shot-quality", "logistic-regression"],
                ),
                ThemeNoteSynthesis(
                    theme_name="Packing Rate & Passing Value",
                    core_concept="Quantifying how many opponents are bypassed by a forward pass.",
                    formal_model_and_equations="""$$\\text{Packing}(p) = \\sum_{k \\in \\text{Defenders}} \\mathbb{I}(\\text{Bypassed}(k, p))$$""",
                    analytical_takeaways=["Passing through lines yields higher goal equity than horizontal possession."],
                    source_evidence=["> \"Packing rate measures how many opponents are taken out of the game.\"\n>\n> *— Kindle Location 340*"],
                    suggested_tags=["packing", "passing"],
                ),
            ],
        )

        pipeline = SyncPipeline(vault, index, router, mock_llm, synthesizer=mock_synth)

        highlights = [
            KindleHighlight(
                book_title="Football Hackers",
                clean_title="Football Hackers",
                author="Christoph Biermann",
                location="120",
                text="Goals are rare events; assessing shot quality gives a much clearer signal of performance.",
            ),
            KindleHighlight(
                book_title="Football Hackers",
                clean_title="Football Hackers",
                author="Christoph Biermann",
                location="340",
                text="Packing rate measures how many opponents are taken out of the game.",
            ),
        ]

        summary = pipeline.sync_highlights(highlights, synthesize_concept_notes=True)

        assert summary.total_processed == 2
        assert summary.themes_generated == 2
        assert summary.llm_calls_made == 1
        assert len(summary.notes_created) == 3  # 00 - Overview.md + 2 Theme Notes

        # Verify folder structure
        book_folder = vault.get_book_folder_path("Football Hackers")
        assert os.path.exists(os.path.join(book_folder, "00 - Overview.md"))
        assert os.path.exists(os.path.join(book_folder, "Expected Goals & Shot Quality Models.md"))
        assert os.path.exists(os.path.join(book_folder, "Packing Rate & Passing Value.md"))

        # Verify LaTeX content
        with open(os.path.join(book_folder, "Expected Goals & Shot Quality Models.md"), "r") as f:
            content = f.read()
        assert "\\text{xG}_i" in content
        assert "$\\sigma(z) = \\frac{1}{1 + e^{-z}}$" in content
        assert "## Formal Model & Equations" in content

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

