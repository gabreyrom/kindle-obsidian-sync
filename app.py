"""Kindle to Obsidian Sync - Streamlit Web Application.

A local web application that syncs Kindle book highlights into an Obsidian vault
using a cost-effective hybrid vector deduplication pipeline (ChromaDB + Gemini Flash)
with scoped directory hierarchies, LaTeX equation extraction, and free-tier rate limiting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
import pandas as pd
import streamlit as st

from core.llm_merger import LLMMerger
from core.models import ActionType, KindleHighlight, SyncItemAction, SyncSummary
from core.parser import group_highlights_by_book, parse_clippings_content, parse_clippings_file
from core.pipeline import SyncPipeline
from core.rate_limiter import RateLimiter
from core.router import DEFAULT_DUPLICATE_THRESHOLD, DEFAULT_NOVEL_THRESHOLD, VectorRouter
from core.synthesizer import BookSynthesizer
from core.vault import VaultManager
from core.vector_index import VaultVectorIndex

load_dotenv()

st.set_page_config(
    page_title="Kindle to Obsidian Sync",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean UI styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #4f46e5;
    }
    .stProgress > div > div > div > div {
        background-color: #4f46e5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "parsed_highlights" not in st.session_state:
    st.session_state.parsed_highlights = []
if "grouped_books" not in st.session_state:
    st.session_state.grouped_books = {}
if "sync_summary" not in st.session_state:
    st.session_state.sync_summary = None
if "selected_books" not in st.session_state:
    st.session_state.selected_books = []


# ==========================================
# Sidebar: Settings & Thresholds
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("1. Obsidian Vault")
    env_vault = os.environ.get("DEFAULT_VAULT_PATH", "")
    vault_path_input = st.text_input(
        "Vault Base Directory",
        value=env_vault,
        placeholder="/Users/username/Obsidian/MyVault",
        help="Absolute path to the base of your Obsidian vault.",
    )

    vault_subfolder = st.text_input(
        "Subdirectory relative to vault",
        value="Books",
        help="Target folder in your vault where dedicated book folders will be created.",
    )

    if vault_path_input:
        if os.path.isdir(vault_path_input):
            st.success(f"✓ Valid vault: `{Path(vault_path_input).name}`", icon="✅")
        else:
            st.warning("Vault path does not exist yet. It will be created on sync.", icon="⚠️")

    st.divider()

    st.subheader("2. Gemini API (Free Tier Safe)")
    env_api_key = os.environ.get("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=env_api_key,
        type="password",
        help="Required for synthesizing conceptual themes and delta merging without quota exhaustion.",
    )

    model_options = [
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
    ]
    selected_model = st.selectbox(
        "Gemini Model",
        options=model_options,
        index=0,
        help="Gemini Flash models offer high rate limits and fast structured JSON generation.",
    )

    free_tier_mode = st.toggle(
        "Free Tier Quota Management",
        value=True,
        help="Enforces >= 4.2s pacing and exponential backoff retry to guarantee 0 quota exceedance on Google's Free Tier.",
    )

    st.divider()

    st.subheader("3. Vector Gating Thresholds")
    st.caption("3-Tier Cosine Similarity Gating Rules")

    dup_thresh = st.slider(
        "Duplicate Cutoff (S_max ≥)",
        min_value=0.70,
        max_value=0.98,
        value=DEFAULT_DUPLICATE_THRESHOLD,
        step=0.01,
        help="Highlights with similarity above this are discarded as redundant without calling the LLM.",
    )

    novel_thresh = st.slider(
        "Novel Cutoff (S_max <)",
        min_value=0.40,
        max_value=0.80,
        value=DEFAULT_NOVEL_THRESHOLD,
        step=0.01,
        help="Highlights with similarity below this are directly appended without calling the LLM.",
    )

    if novel_thresh >= dup_thresh:
        st.error("Error: Novel threshold must be strictly less than Duplicate threshold.")

    st.info(
        f"**Cost Gating Summary**:\n"
        f"- `S ≥ {dup_thresh:.2f}`: Discard duplicate (Free)\n"
        f"- `{novel_thresh:.2f} ≤ S < {dup_thresh:.2f}`: Call Gemini Flash\n"
        f"- `S < {novel_thresh:.2f}`: Direct insert (Free)"
    )


# ==========================================
# Main Header
# ==========================================
st.markdown('<div class="main-title">📚 Kindle to Obsidian Sync</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Autonomous Knowledge Base Sync with <b>Scoped Vector Deduplication</b> and '
    '<b>Thematic LaTeX Synthesis</b>.'
    '</div>',
    unsafe_allow_html=True,
)

# ==========================================
# Step 1: Input Kindle Clippings
# ==========================================
st.subheader("Step 1: Provide Kindle Clippings")

input_tab1, input_tab2, input_tab3 = st.tabs(["📁 Upload File", "🖥️ File Path", "💡 Demo Sample"])

with input_tab1:
    uploaded_file = st.file_uploader(
        "Upload your 'My Clippings.txt' file from your Kindle device",
        type=["txt"],
        help="Connect your Kindle via USB and locate 'My Clippings.txt' inside the 'documents' folder.",
    )
    if uploaded_file is not None:
        try:
            content = uploaded_file.getvalue().decode("utf-8-sig", errors="replace")
            highlights = parse_clippings_content(content)
            st.session_state.parsed_highlights = highlights
            st.session_state.grouped_books = group_highlights_by_book(highlights)
            st.success(f"Parsed {len(highlights)} highlights across {len(st.session_state.grouped_books)} books from uploaded file.")
        except Exception as e:
            st.error(f"Error parsing uploaded file: {e}")

with input_tab2:
    col_path, col_btn = st.columns([4, 1])
    with col_path:
        clippings_path = st.text_input(
            "Local file path to 'My Clippings.txt'",
            placeholder="/Volumes/Kindle/documents/My Clippings.txt",
        )
    with col_btn:
        st.write("")
        st.write("")
        if st.button("Parse File", width="stretch"):
            if clippings_path and os.path.exists(clippings_path):
                highlights = parse_clippings_file(clippings_path)
                st.session_state.parsed_highlights = highlights
                st.session_state.grouped_books = group_highlights_by_book(highlights)
                st.success(f"Successfully parsed {len(highlights)} highlights across {len(st.session_state.grouped_books)} books.")
            else:
                st.error("File not found at specified path.")

with input_tab3:
    st.write("Load a representative sample dataset with quantitative sports analytics and psychology highlights.")
    if st.button("Load Sample Clippings Dataset"):
        sample_text = """Football Hackers: The Science and Art of a Data Revolution (Christoph Biermann)
- Your Highlight on page 42 | location 520-525 | Added on Sunday, October 4, 2020 1:23:45 PM

Expected goals measures the probability of a shot resulting in a goal based on historical shot trajectory, distance, and angle. Goals are rare events; assessing shot quality gives a much clearer signal of performance.
==========
Football Hackers: The Science and Art of a Data Revolution (Christoph Biermann)
- Your Highlight on page 48 | location 610-614 | Added on Sunday, October 4, 2020 2:00:10 PM

Post-shot expected goals, or xGOT, evaluates the shot trajectory after the ball leaves the striker's foot. It isolates goalkeeper shot-stopping ability by accounting for shot speed and placement.
==========
Football Hackers: The Science and Art of a Data Revolution (Christoph Biermann)
- Your Highlight on page 85 | location 1020-1024 | Added on Monday, November 15, 2021 9:14:22 AM

Packing rate measures how many opponents are taken out of the game with a single forward pass. Taking six defenders out with one vertical ball yields immense goal equity.
==========
Thinking, Fast and Slow (Daniel Kahneman)
- Your Highlight on page 12 | location 174-175 | Added on Sunday, October 4, 2020 1:23:45 PM

A reliable way to make people believe in falsehoods is frequent repetition, because familiarity is not easily distinguished from truth.
==========
Thinking, Fast and Slow (Daniel Kahneman)
- Your Highlight on page 15 | location 210-212 | Added on Sunday, October 4, 2020 2:00:10 PM

System 1 operates automatically and quickly, with little or no effort and no sense of voluntary control.
==========
"""
        highlights = parse_clippings_content(sample_text)
        st.session_state.parsed_highlights = highlights
        st.session_state.grouped_books = group_highlights_by_book(highlights)
        st.success(f"Loaded demo dataset with {len(highlights)} highlights across {len(st.session_state.grouped_books)} books.")


# ==========================================
# Step 2: Select Book(s) to Process
# ==========================================
if st.session_state.grouped_books:
    st.divider()
    st.subheader("Step 2: Select Book(s) to Sync")

    all_books = list(st.session_state.grouped_books.keys())
    book_options = [f"{b} ({len(st.session_state.grouped_books[b])} highlights)" for b in all_books]
    title_map = {f"{b} ({len(st.session_state.grouped_books[b])} highlights)": b for b in all_books}

    selected_options = st.multiselect(
        "Choose book(s) to sync to your Obsidian vault:",
        options=book_options,
        default=[book_options[0]] if book_options else [],
    )

    st.session_state.selected_books = [title_map[opt] for opt in selected_options]

    # Collect active highlights
    active_highlights: List[KindleHighlight] = []
    for b in st.session_state.selected_books:
        active_highlights.extend(st.session_state.grouped_books.get(b, []))

    if active_highlights:
        with st.expander(f"🔍 Preview {len(active_highlights)} Highlights for Selected Book(s)"):
            for idx, h in enumerate(active_highlights, 1):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"**{idx}. [{h.clean_title}]**")
                    st.markdown(f"> *\"{h.text}\"*")
                with col2:
                    st.caption(f"📍 {h.location or 'N/A'}")
                    st.caption(f"🗓️ {h.timestamp or 'N/A'}")
                st.divider()

    # ==========================================
    # Step 3: Process and Sync
    # ==========================================
    st.divider()
    st.subheader("Step 3: Process & Sync to Obsidian Vault")

    col_mode1, col_mode2 = st.columns([3, 2])
    with col_mode1:
        synthesize_mode = st.checkbox(
            "Synthesize Concept Notes with LaTeX (Recommended)",
            value=True,
            help="Ingests highlights in a single pass to cluster into 3 to 6 atomic conceptual themes with Obsidian-compatible LaTeX formulas ($inline$, $$display$$) and an Overview TOC.",
        )

    col_sync_btn, col_info = st.columns([2, 3])
    with col_sync_btn:
        start_sync = st.button(
            "🚀 Process and Sync to Vault",
            type="primary",
            width="stretch",
            disabled=len(active_highlights) == 0,
        )

    with col_info:
        if not vault_path_input:
            st.warning("Please configure your Obsidian Vault Base Directory in the sidebar.")
        elif not api_key_input:
            st.warning("Gemini API Key is empty in sidebar. LLM synthesis will fall back to local formatting.")

    if start_sync:
        if not vault_path_input:
            st.error("Cannot proceed: Please enter an Obsidian Vault Base Directory in the sidebar.")
            st.stop()

        vault_mgr = VaultManager(vault_path=vault_path_input, default_folder=vault_subfolder)
        vector_idx = VaultVectorIndex()
        router = VectorRouter(duplicate_threshold=dup_thresh, novel_threshold=novel_thresh)

        rate_limit_interval = 4.2 if free_tier_mode else 0.0
        rate_limiter = RateLimiter(min_interval_seconds=rate_limit_interval)

        cooldown_container = st.empty()
        status_text = st.empty()

        def rate_limit_cooldown_callback(msg: str):
            cooldown_container.warning(f"⏳ **Rate Limiter Pacing**: {msg}")

        llm_engine = LLMMerger(
            api_key=api_key_input,
            model=selected_model,
            min_request_interval=rate_limit_interval,
            rate_limiter=rate_limiter,
            status_callback=rate_limit_cooldown_callback,
        )

        synthesizer_engine = BookSynthesizer(
            api_key=api_key_input,
            model=selected_model,
            min_request_interval=rate_limit_interval,
            rate_limiter=rate_limiter,
            status_callback=rate_limit_cooldown_callback,
        )

        pipeline = SyncPipeline(
            vault_manager=vault_mgr,
            vector_index=vector_idx,
            router=router,
            llm_merger=llm_engine,
            synthesizer=synthesizer_engine,
        )

        progress_bar = st.progress(0, text="Initializing scoped book pipeline...")

        def update_progress(current: int, total: int, item: SyncItemAction):
            pct = int((current / total) * 100)
            status_text.markdown(
                f"**Processing {current}/{total}**: `{item.highlight.clean_title}` | "
                f"Action: `{item.final_action.value}` (Similarity: `{item.similarity_score:.3f}`)"
            )
            progress_bar.progress(pct)

        with st.spinner("Executing scoped deduplication & thematic LaTeX synthesis..."):
            summary = pipeline.sync_highlights(
                highlights=active_highlights,
                progress_callback=update_progress,
                synthesize_concept_notes=synthesize_mode,
                status_callback=lambda s: status_text.info(f"🔄 {s}"),
            )
            st.session_state.sync_summary = summary
            cooldown_container.empty()
            progress_bar.progress(100, text="Sync complete!")
            st.success("✅ Sync completed successfully!")


# ==========================================
# Step 4: Live Telemetry & Action Summary
# ==========================================
if st.session_state.sync_summary:
    summary: SyncSummary = st.session_state.sync_summary
    st.divider()
    st.subheader("📊 Execution Telemetry & Status Cards")

    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Total Highlights", summary.total_processed)
    with m_col2:
        st.metric("Duplicates Skipped", summary.duplicates_discarded, delta="Zero LLM Cost", delta_color="normal")
    with m_col3:
        st.metric("Themes Generated", summary.themes_generated, delta="LaTeX Math Notes", delta_color="normal")
    with m_col4:
        st.metric("API Calls Made", summary.llm_calls_made, delta="Gemini Flash", delta_color="off")
    with m_col5:
        st.metric("Cost Reduction", f"{summary.llm_calls_saved_percent:.1f}%", delta="Calls Saved", delta_color="normal")

    st.subheader("Detailed Decision Breakdown")

    # Build DataFrame for interactive table
    table_rows = []
    for it in summary.items:
        table_rows.append({
            "Book": it.highlight.clean_title,
            "Highlight Excerpt": it.highlight.text[:90] + ("..." if len(it.highlight.text) > 90 else ""),
            "Max Cosine Sim": f"{it.similarity_score:.3f}",
            "Action Taken": it.final_action.value,
            "LLM Invoked": "🧠 Yes" if it.llm_invoked else "⚡ No",
            "Target Note": Path(it.target_file).name,
            "Reasoning": it.reasoning,
        })

    df = pd.DataFrame(table_rows)
    st.dataframe(df, width="stretch")

    # Note Viewer: Preview the updated Obsidian Markdown file
    if summary.notes_created or (vault_path_input and os.path.exists(vault_path_input)):
        st.subheader("📄 Obsidian Vault Note Preview (with LaTeX Math)")
        existing_notes = [f for f in summary.notes_created if os.path.exists(f)]
        if not existing_notes:
            existing_notes = list(dict.fromkeys(it.target_file for it in summary.items if os.path.exists(it.target_file)))

        if existing_notes:
            selected_preview_file = st.selectbox(
                "Select generated note to preview:",
                options=existing_notes,
                format_func=lambda x: f"{Path(x).parent.name} / {Path(x).name}",
            )
            if selected_preview_file and os.path.exists(selected_preview_file):
                with open(selected_preview_file, "r", encoding="utf-8") as f:
                    note_content = f.read()

                tab_preview, tab_raw = st.tabs(["Formatted Markdown (Rendered LaTeX)", "Raw Markdown Source"])
                with tab_preview:
                    st.markdown(note_content)
                with tab_raw:
                    st.code(note_content, language="markdown")
