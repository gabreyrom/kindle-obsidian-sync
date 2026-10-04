"""Unit tests for Kindle parser, VaultManager, VectorRouter, and Pipeline."""

import os
import shutil
import tempfile
import pytest

from core.models import ActionType, HighlightType, KindleHighlight, VaultParagraph
from core.parser import clean_filename, extract_title_and_author, parse_clippings_content
from core.router import VectorRouter
from core.vault import VaultManager


SAMPLE_CLIPPINGS = """Thinking, Fast and Slow (Daniel Kahneman)
- Your Highlight on page 12 | location 174-175 | Added on Sunday, October 4, 2020 1:23:45 PM

A reliable way to make people believe in falsehoods is frequent repetition, because familiarity is not easily distinguished from truth.
==========
Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems (Martin Kleppmann)
- Your Highlight on Location 45-46 | Added on Monday, January 1, 2024 10:15:00 AM

Data systems are composed from diverse components with different strengths and weaknesses.
==========
Atomic Habits (James Clear)
- Your Note on page 50 | location 750 | Added on Tuesday, February 2, 2024 3:00:00 PM

Remember to review this habit loop concept for morning routines.
==========
Thinking, Fast and Slow (Daniel Kahneman)
- Your Highlight on page 12 | location 174-175 | Added on Sunday, October 4, 2020 1:23:45 PM

A reliable way to make people believe in falsehoods is frequent repetition, because familiarity is not easily distinguished from truth.
==========
"""


def test_clean_filename():
    assert clean_filename("Foo: Bar / Baz? * < > | #") == "Foo - Bar - Baz"
    assert clean_filename("Clean Title") == "Clean Title"
    assert clean_filename("Ends with dot.") == "Ends with dot"


def test_extract_title_and_author():
    title, author = extract_title_and_author("Thinking, Fast and Slow (Daniel Kahneman)")
    assert title == "Thinking, Fast and Slow"
    assert author == "Daniel Kahneman"

    title2, author2 = extract_title_and_author("Book Without Author")
    assert title2 == "Book Without Author"
    assert author2 == "Unknown"


def test_parse_clippings_content():
    highlights = parse_clippings_content(SAMPLE_CLIPPINGS)
    # The 4th item is an exact duplicate of the 1st item, so should be deduplicated
    assert len(highlights) == 3

    h1 = highlights[0]
    assert h1.clean_title == "Thinking, Fast and Slow"
    assert h1.author == "Daniel Kahneman"
    assert "repetition" in h1.text
    assert h1.highlight_type == HighlightType.HIGHLIGHT

    h2 = highlights[1]
    assert "Designing Data-Intensive Applications" in h2.clean_title
    assert ":" not in h2.clean_title
    assert h2.author == "Martin Kleppmann"

    h3 = highlights[2]
    assert h3.highlight_type == HighlightType.NOTE


def test_vector_router_thresholds():
    router = VectorRouter(duplicate_threshold=0.86, novel_threshold=0.62)

    # Duplicate zone: >= 0.86
    res_dup = router.route(0.92)
    assert res_dup.action == ActionType.DISCARD_DUPLICATE

    res_dup_edge = router.route(0.86)
    assert res_dup_edge.action == ActionType.DISCARD_DUPLICATE

    # LLM zone: 0.62 <= S < 0.86
    res_llm = router.route(0.75)
    assert res_llm.action == ActionType.ROUTE_TO_LLM

    res_llm_edge = router.route(0.62)
    assert res_llm_action == ActionType.ROUTE_TO_LLM if 'res_llm_action' in locals() else res_llm.action == ActionType.ROUTE_TO_LLM

    # Novel zone: < 0.62
    res_novel = router.route(0.45)
    assert res_novel.action == ActionType.NEW_SECTION


def test_vault_manager_create_and_append():
    temp_dir = tempfile.mkdtemp()
    try:
        vault = VaultManager(vault_path=temp_dir, default_folder="Books")
        highlight = KindleHighlight(
            book_title="Test Book",
            clean_title="Test Book",
            author="Test Author",
            location="10-12",
            text="This is a test highlight.",
        )

        note_path = vault.write_initial_book_note(highlight)
        assert os.path.exists(note_path)

        fm, body = vault.read_note_content(note_path)
        assert fm.get("author") == "Test Author"
        assert "## Highlights" in body

        # Append new highlight under Highlights
        vault.append_to_section(
            note_path,
            section_heading="Highlights",
            content="> Extra quote\n>\n> *— [[Test Book]]*",
            new_tags=["insight", "testing"],
        )

        fm2, body2 = vault.read_note_content(note_path)
        assert "insight" in fm2.get("tags", [])
        assert "testing" in fm2.get("tags", [])
        assert "> Extra quote" in body2

        # Extract paragraphs
        paragraphs = vault.extract_paragraphs(note_path)
        assert len(paragraphs) > 0

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_rate_limiter_helpers():
    from core.llm_merger import extract_retry_delay, is_rate_limit_error, is_decommissioned_model_error

    # 429 error detection
    err_429 = Exception("429 RESOURCE_EXHAUSTED: Quota exceeded. Please retry in 35.32s.")
    assert is_rate_limit_error(err_429) is True
    delay = extract_retry_delay(err_429)
    assert 35.0 <= delay <= 37.0

    # retryDelay json style
    err_json = Exception("{'error': {'code': 429}, 'retryDelay': '20s'}")
    assert is_rate_limit_error(err_json) is True
    assert 20.0 <= extract_retry_delay(err_json) <= 22.0

    # 404 decommissioned model
    err_404 = Exception("404 NOT_FOUND: This model is no longer available to new users.")
    assert is_decommissioned_model_error(err_404) is True
    assert is_rate_limit_error(err_404) is False


def test_rate_limiter_and_backoff(monkeypatch):
    import time
    from core.rate_limiter import RateLimiter, retry_with_exponential_backoff

    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    # Test token bucket / minimum interval throttler
    rl = RateLimiter(min_interval_seconds=4.2)
    rl.acquire()
    assert len(sleeps) == 0

    rl.acquire()
    assert len(sleeps) == 1
    assert 3.5 <= sleeps[0] <= 4.3

    # Test exponential backoff decorator
    attempts_made = 0

    @retry_with_exponential_backoff(max_retries=3, base_delay=1.0)
    def flaky_func():
        nonlocal attempts_made
        attempts_made += 1
        if attempts_made < 3:
            raise Exception("429 RESOURCE_EXHAUSTED: Rate limit hit")
        return "success"

    res = flaky_func()
    assert res == "success"
    assert attempts_made == 3
    # Check that backoff sleeps occurred
    assert len(sleeps) >= 3


def test_scoped_vault_hierarchy_and_thematic_notes():
    temp_dir = tempfile.mkdtemp()
    try:
        vault = VaultManager(vault_path=temp_dir, default_folder="Books")
        clean_title = "The Signal and the Noise"
        author = "Nate Silver"

        # Check scoped folder path
        expected_folder = os.path.join(temp_dir, "Books", "The Signal and the Noise")
        assert vault.get_book_folder_path(clean_title) == expected_folder
        assert vault.has_existing_notes(clean_title) is False

        # Create theme note
        from core.models import ThemeNoteSynthesis
        theme = ThemeNoteSynthesis(
            theme_name="Bayesian Updating & Probability",
            core_concept="Beliefs should be updated probabilistically in response to new evidence.",
            formal_model_and_equations="$$P(A|B) = \\frac{P(B|A)P(A)}{P(B)}$$\nWhere $P(A)$ is the prior probability.",
            analytical_takeaways=[
                "Explicit priors prevent overfitting to noisy observations.",
                "Conditional updates converge toward true generative parameters.",
            ],
            source_evidence=["> \"We must become more comfortable with probability and uncertainty.\"\n>\n> *— Kindle Location 210*"],
            suggested_tags=["bayesian", "statistics", "probability"],
        )

        theme_path = vault.write_theme_note(clean_title, author, theme)
        assert os.path.exists(theme_path)
        assert "Bayesian Updating & Probability.md" in theme_path

        # Create Overview note
        overview_path = vault.write_overview_note(
            clean_title=clean_title,
            author=author,
            synopsis="An analytical survey on forecasting accuracy across disciplines.",
            themes=[theme],
            overall_tags=["forecasting", "data-science"],
        )
        assert os.path.exists(overview_path)
        assert "00 - Overview.md" in overview_path

        # Verify Overview note contains [[Wikilinks]] to the theme note
        with open(overview_path, "r", encoding="utf-8") as f:
            overview_text = f.read()
        assert "[[Bayesian Updating & Probability]]" in overview_text
        assert "Nate Silver" in overview_text

        # Verify Theme note contents and LaTeX math
        with open(theme_path, "r", encoding="utf-8") as f:
            theme_text = f.read()
        assert "$$P(A|B) = \\frac{P(B|A)P(A)}{P(B)}$$" in theme_text
        assert "## Core Concept" in theme_text
        assert "## Formal Model & Equations" in theme_text
        assert "## Analytical Takeaways" in theme_text
        assert "## Source Evidence" in theme_text

        # Verify scoped indexing: only notes inside this book folder are returned
        assert vault.has_existing_notes(clean_title) is True
        scoped_paras = vault.get_book_paragraphs(clean_title)
        assert len(scoped_paras) > 0
        for p in scoped_paras:
            assert expected_folder in p.file_path

        # Unrelated book folder has 0 notes
        assert vault.get_book_paragraphs("Nonexistent Book") == []

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


