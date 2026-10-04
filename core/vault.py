"""Obsidian Vault I/O Handler.

Handles scoped directory hierarchies, reading, chunking, frontmatter management,
and writing of multi-note conceptual themes in an Obsidian vault.
"""

from __future__ import annotations

from datetime import datetime
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import yaml

from core.models import (
    ActionType,
    KindleHighlight,
    LLMMergeResponse,
    ThemeNoteSynthesis,
    VaultParagraph,
)
from core.parser import clean_filename

FRONTMATTER_REGEX = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class VaultManager:
    """Manages reading and writing notes inside a scoped Obsidian book directory."""

    def __init__(self, vault_path: str, default_folder: str = "Books"):
        self.vault_path = os.path.abspath(os.path.expanduser(vault_path))
        self.default_folder = default_folder.strip().strip("/\\")

    @property
    def target_dir(self) -> str:
        """The absolute base directory where book folders are stored."""
        if self.default_folder:
            return os.path.join(self.vault_path, self.default_folder)
        return self.vault_path

    def get_book_folder_path(self, clean_title: str) -> str:
        """Return scoped directory: {vault_path}/{target_subdirectory}/{Sanitized_Book_Title}/"""
        sanitized = clean_filename(clean_title)
        return os.path.join(self.target_dir, sanitized)

    def ensure_book_folder(self, clean_title: str) -> str:
        """Ensure the dedicated book folder exists on disk."""
        folder = self.get_book_folder_path(clean_title)
        os.makedirs(folder, exist_ok=True)
        return folder

    def has_existing_notes(self, clean_title: str) -> bool:
        """Check if the book folder exists and contains any .md files."""
        folder = self.get_book_folder_path(clean_title)
        if not os.path.isdir(folder):
            return False
        md_files = [f for f in os.listdir(folder) if f.endswith(".md")]
        return len(md_files) > 0

    def get_book_note_path(self, clean_title: str) -> str:
        """Return the primary path for a book's overview note."""
        folder = self.get_book_folder_path(clean_title)
        return os.path.join(folder, "00 - Overview.md")

    def read_note_content(self, file_path: str) -> Tuple[Dict[str, Any], str]:
        """Read an Obsidian markdown file, splitting it into YAML frontmatter dict and body text."""
        if not os.path.exists(file_path):
            return {}, ""

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        match = FRONTMATTER_REGEX.match(content)
        if match:
            fm_text = match.group(1)
            body = content[match.end():]
            try:
                frontmatter = yaml.safe_load(fm_text) or {}
                if not isinstance(frontmatter, dict):
                    frontmatter = {}
            except Exception:
                frontmatter = {}
            return frontmatter, body

        return {}, content

    def extract_paragraphs(self, file_path: str) -> List[VaultParagraph]:
        """Extract meaningful paragraphs from a note for vector indexing."""
        if not os.path.exists(file_path):
            return []

        _, body = self.read_note_content(file_path)
        rel_path = os.path.relpath(file_path, self.vault_path)
        note_title = os.path.splitext(os.path.basename(file_path))[0]

        paragraphs: List[VaultParagraph] = []
        current_heading: Optional[str] = None

        blocks = re.split(r"\n\s*\n", body)
        idx = 0
        for block in blocks:
            text = block.strip()
            if not text:
                continue

            heading_match = re.match(r"^(#{1,6})\s+(.+)$", text)
            if heading_match:
                current_heading = heading_match.group(2).strip()
                lines = text.splitlines()
                if len(lines) > 1:
                    text = "\n".join(lines[1:]).strip()
                else:
                    continue

            if text in ["---", "***", "___"] or len(text) < 15:
                continue

            paragraphs.append(
                VaultParagraph(
                    file_path=file_path,
                    rel_path=rel_path,
                    note_title=note_title,
                    section_heading=current_heading,
                    paragraph_index=idx,
                    text=text,
                )
            )
            idx += 1

        return paragraphs

    def get_book_paragraphs(self, clean_title: str) -> List[VaultParagraph]:
        """Scoped Indexing Constraint:
        ChromaDB and vector deduplication MUST only index .md files located
        inside this specific book folder: {vault_path}/{target_subdirectory}/{Sanitized_Book_Title}/.
        Do NOT scan or index the rest of the Obsidian vault.
        If the book folder does not exist or has no markdown files, returns empty list.
        """
        folder = self.get_book_folder_path(clean_title)
        if not os.path.isdir(folder):
            return []

        paragraphs: List[VaultParagraph] = []
        for file in sorted(os.listdir(folder)):
            if file.endswith(".md"):
                file_path = os.path.join(folder, file)
                paragraphs.extend(self.extract_paragraphs(file_path))

        return paragraphs

    def get_existing_vault_paragraphs(self, specific_book_title: Optional[str] = None) -> List[VaultParagraph]:
        """Enforces scoped indexing: only scans the specific book folder if provided."""
        if specific_book_title:
            return self.get_book_paragraphs(specific_book_title)
        return []

    def format_frontmatter(self, meta: Dict[str, Any]) -> str:
        """Serialize a dict into valid YAML frontmatter block."""
        fm_yaml = yaml.dump(
            meta,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ).strip()
        return f"---\n{fm_yaml}\n---\n\n"

    def write_overview_note(
        self,
        clean_title: str,
        author: str,
        synopsis: str,
        themes: List[ThemeNoteSynthesis],
        overall_tags: Optional[List[str]] = None,
    ) -> str:
        """Write '00 - Overview.md' containing synopsis, metadata, and wikilink Table of Contents."""
        folder = self.ensure_book_folder(clean_title)
        note_path = os.path.join(folder, "00 - Overview.md")
        now_iso = datetime.now().isoformat(timespec="seconds")

        tags = ["book-overview", "kindle-sync"]
        if overall_tags:
            tags.extend(overall_tags)
        clean_tags = sorted(list(set(re.sub(r"[^\w/-]", "", t.lower().replace(" ", "-")) for t in tags if t)))

        frontmatter = {
            "tags": clean_tags,
            "date_synced": now_iso,
            "source_book": clean_title,
            "author": author,
            "total_themes": len(themes),
        }

        # Build Markdown Table of Contents with [[Wikilinks]]
        toc_rows: List[str] = []
        for theme in themes:
            clean_theme_name = clean_filename(theme.theme_name)
            # Escape pipes in core concept summary
            safe_concept = theme.core_concept.replace("|", "\\|").replace("\n", " ").strip()
            toc_rows.append(f"| [[{clean_theme_name}]] | {safe_concept} |")

        toc_table = (
            "| Conceptual Theme Note | Theoretical Summary |\n"
            "| :--- | :--- |\n"
            + "\n".join(toc_rows)
        )

        body = (
            f"# {clean_title} — Overview\n\n"
            f"> **Author**: {author}  \n"
            f"> **Date Synced**: {now_iso}  \n"
            f"> **Target Folder**: `{os.path.basename(folder)}`  \n\n"
            f"---\n\n"
            f"## Synopsis\n\n"
            f"{synopsis.strip()}\n\n"
            f"---\n\n"
            f"## Conceptual Themes & Table of Contents\n\n"
            f"{toc_table}\n\n"
            f"---\n"
            f"*Autonomous Knowledge Base generated via Kindle-to-Obsidian Engine*\n"
        )

        full_content = self.format_frontmatter(frontmatter) + body
        with open(note_path, "w", encoding="utf-8") as f:
            f.write(full_content)

        return note_path

    def write_theme_note(
        self,
        clean_title: str,
        author: str,
        theme: ThemeNoteSynthesis,
    ) -> str:
        """Write '{Theme_Name}.md' matching the rigorous Theme Note Schema."""
        folder = self.ensure_book_folder(clean_title)
        clean_theme_name = clean_filename(theme.theme_name)
        note_path = os.path.join(folder, f"{clean_theme_name}.md")
        now_iso = datetime.now().isoformat(timespec="seconds")

        tags = ["concept", "kindle-highlight"]
        if theme.suggested_tags:
            tags.extend(theme.suggested_tags)
        clean_tags = sorted(list(set(re.sub(r"[^\w/-]", "", t.lower().replace(" ", "-")) for t in tags if t)))

        frontmatter = {
            "tags": clean_tags,
            "source_book": f"[[00 - Overview|{clean_title}]]",
            "author": author,
            "date_synced": now_iso,
        }

        # Format analytical takeaways
        takeaways_md = "\n".join(f"- {item.strip()}" for item in theme.analytical_takeaways if item.strip())
        if not takeaways_md:
            takeaways_md = "- Analytical synthesis pending further domain highlights."

        # Format source evidence blockquotes
        evidence_blocks: List[str] = []
        for quote in theme.source_evidence:
            q_clean = quote.strip()
            if not q_clean:
                continue
            if not q_clean.startswith(">"):
                q_clean = f"> {q_clean}"
            evidence_blocks.append(q_clean)
        evidence_md = "\n\n".join(evidence_blocks) if evidence_blocks else "> *No verbatim citations recorded for this theme.*"

        body = (
            f"# {theme.theme_name.strip()}\n\n"
            f"## Core Concept\n"
            f"{theme.core_concept.strip()}\n\n"
            f"## Formal Model & Equations\n"
            f"{theme.formal_model_and_equations.strip()}\n\n"
            f"## Analytical Takeaways\n"
            f"{takeaways_md}\n\n"
            f"## Source Evidence\n"
            f"{evidence_md}\n\n"
            f"---\n"
            f"*Part of [[00 - Overview|{clean_title}]] Knowledge Base*\n"
        )

        full_content = self.format_frontmatter(frontmatter) + body
        with open(note_path, "w", encoding="utf-8") as f:
            f.write(full_content)

        return note_path

    def append_to_section(
        self,
        file_path: str,
        section_heading: Optional[str],
        content: str,
        new_tags: Optional[List[str]] = None,
    ) -> None:
        """Append content under a given section, or create section at the end of the note."""
        fm, body = self.read_note_content(file_path)

        existing_tags = set(fm.get("tags", []))
        if new_tags:
            for t in new_tags:
                clean_tag = re.sub(r"[^\w/-]", "", t.strip().lower().replace(" ", "-"))
                if clean_tag:
                    existing_tags.add(clean_tag)
        fm["tags"] = sorted(list(existing_tags))
        fm["date_synced"] = datetime.now().isoformat(timespec="seconds")

        heading_title = (section_heading or "Highlights").strip()
        target_heading_pattern = rf"(?m)^##\s+{re.escape(heading_title)}\s*$"

        if re.search(target_heading_pattern, body):
            parts = re.split(target_heading_pattern, body, maxsplit=1)
            next_heading = re.search(r"(?m)^#{1,6}\s+", parts[1])
            if next_heading:
                insert_pos = next_heading.start()
                updated_body = (
                    parts[0]
                    + f"## {heading_title}\n"
                    + parts[1][:insert_pos].rstrip()
                    + f"\n\n{content}\n\n"
                    + parts[1][insert_pos:]
                )
            else:
                updated_body = (
                    parts[0]
                    + f"## {heading_title}\n"
                    + parts[1].rstrip()
                    + f"\n\n{content}\n"
                )
        else:
            updated_body = body.rstrip() + f"\n\n## {heading_title}\n\n{content}\n"

        full_content = self.format_frontmatter(fm) + updated_body.lstrip()
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_content)

    def write_initial_book_note(self, highlight: KindleHighlight) -> str:
        """Create a default overview note if raw highlights mode is used."""
        folder = self.ensure_book_folder(highlight.clean_title)
        note_path = os.path.join(folder, "00 - Overview.md")
        now_iso = datetime.now().isoformat(timespec="seconds")

        frontmatter = {
            "tags": ["kindle-highlight", "book"],
            "date_synced": now_iso,
            "source_book": highlight.book_title,
            "author": highlight.author,
        }

        body = (
            f"# {highlight.clean_title}\n\n"
            f"> **Author**: {highlight.author}  \n"
            f"> **Last Synced**: {now_iso}  \n\n"
            f"---\n\n"
            f"## Highlights\n\n"
        )

        full_content = self.format_frontmatter(frontmatter) + body
        with open(note_path, "w", encoding="utf-8") as f:
            f.write(full_content)

        return note_path

    def apply_sync_action(
        self,
        highlight: KindleHighlight,
        action: ActionType,
        content_to_insert: str,
        target_section: Optional[str] = None,
        suggested_tags: Optional[List[str]] = None,
    ) -> Tuple[str, str]:
        """Apply incremental routing/merge action inside the scoped book folder."""
        folder = self.ensure_book_folder(highlight.clean_title)
        book_note = self.get_book_note_path(highlight.clean_title)

        if action == ActionType.DISCARD_DUPLICATE:
            return book_note, "Skipped (Duplicate detected)"

        if not os.path.exists(book_note):
            self.write_initial_book_note(highlight)

        if action in (ActionType.MERGE_APPEND, ActionType.NEW_SECTION):
            sec = target_section or ("Key Concepts" if action == ActionType.NEW_SECTION else "Highlights")
            self.append_to_section(book_note, sec, content_to_insert, suggested_tags)
            return book_note, f"Appended under '## {sec}'"

        if action == ActionType.STANDALONE_NOTE:
            concept_title = target_section or f"Concept - {highlight.clean_title[:30]}"
            clean_concept = clean_filename(concept_title)
            note_path = os.path.join(folder, f"{clean_concept}.md")
            now_iso = datetime.now().isoformat(timespec="seconds")

            tags = ["concept", "kindle-highlight"]
            if suggested_tags:
                tags.extend(suggested_tags)

            frontmatter = {
                "tags": sorted(list(set(tags))),
                "date_synced": now_iso,
                "source_book": f"[[00 - Overview|{highlight.clean_title}]]",
                "author": highlight.author,
            }

            body = (
                f"# {clean_concept}\n\n"
                f"{content_to_insert}\n\n"
                f"---\n"
                f"*Source: [[00 - Overview|{highlight.clean_title}]] by {highlight.author}*\n"
            )

            full_content = self.format_frontmatter(frontmatter) + body
            with open(note_path, "w", encoding="utf-8") as f:
                f.write(full_content)

            link_entry = f"- Concept Note: [[{clean_concept}]]"
            self.append_to_section(book_note, "Linked Concepts", link_entry, suggested_tags)
            return note_path, f"Created concept note [[{clean_concept}]]"

        return book_note, "No action"
