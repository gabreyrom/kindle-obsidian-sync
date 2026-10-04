"""Kindle 'My Clippings.txt' parser and title normalizer."""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional, Tuple

from core.models import HighlightType, KindleHighlight

ILLEGAL_CHAR_REGEX = re.compile(r'[\/:*?"<>|#^\[\]]')
MULTIPLE_SPACES_REGEX = re.compile(r'\s+')


def clean_filename(title: str) -> str:
    """Normalize a book title to a safe, valid Obsidian filename and link text.

    Replaces colons with ' - ', replaces slashes with '-', strips invalid
    characters, collapses multiple spaces, and strips trailing punctuation.
    """
    if not title:
        return "Untitled Note"

    # Replace colons often used in subtitles with clean separator
    cleaned = title.replace(":", " - ")
    # Replace slashes with dash
    cleaned = cleaned.replace("/", "-").replace("\\", "-")
    # Remove characters forbidden by filesystems and Obsidian wiki-links
    cleaned = ILLEGAL_CHAR_REGEX.sub("", cleaned)
    # Collapse multiple spaces and trim
    cleaned = MULTIPLE_SPACES_REGEX.sub(" ", cleaned).strip()
    # Strip trailing periods/spaces (problematic on Windows/macOS file paths)
    cleaned = cleaned.strip(". ")

    return cleaned or "Untitled Note"


def extract_title_and_author(first_line: str) -> Tuple[str, str]:
    """Extract book title and author from the first line of a Kindle clipping.

    Standard Kindle format puts the author in the trailing parentheses:
    'Book Title: Subtitle (Author Name)'
    """
    raw = first_line.strip()
    if not raw:
        return "Untitled", "Unknown"

    # Match trailing parentheses for author
    match = re.search(r"^(.*?)\s*\(([^()]+)\)$", raw)
    if match:
        title = match.group(1).strip()
        author = match.group(2).strip()
        # If title was entirely inside parentheses or empty, fallback
        if not title:
            title = author
            author = "Unknown"
        return title, author

    return raw, "Unknown"


def parse_metadata_line(meta_line: str) -> Tuple[HighlightType, str, Optional[str], Optional[str]]:
    """Parse Kindle metadata line containing type, page, location, and timestamp.

    Examples:
    - "- Your Highlight on page 12 | location 174-175 | Added on Sunday, October 4, 2020 1:23:45 PM"
    - "- Your Highlight on Location 45-46 | Added on Monday, January 1, 2024 10:15:00 AM"
    - "- Your Note on Location 120 | Added on..."
    """
    lower_line = meta_line.lower()

    if "note" in lower_line:
        h_type = HighlightType.NOTE
    elif "bookmark" in lower_line:
        h_type = HighlightType.BOOKMARK
    else:
        h_type = HighlightType.HIGHLIGHT

    # Extract page
    page_match = re.search(r"page\s+([0-9]+(?:-[0-9]+)?)", meta_line, re.IGNORECASE)
    page = page_match.group(1) if page_match else None

    # Extract location
    loc_match = re.search(r"location\s+([0-9]+(?:-[0-9]+)?)", meta_line, re.IGNORECASE)
    location = loc_match.group(1) if loc_match else ""
    if not location and page:
        location = f"Page {page}"

    # Extract added timestamp
    time_match = re.search(r"Added on\s+(.+)$", meta_line, re.IGNORECASE)
    timestamp = time_match.group(1).strip() if time_match else None

    return h_type, location, page, timestamp


def parse_clippings_content(content: str) -> List[KindleHighlight]:
    """Parse raw content of a 'My Clippings.txt' file into a list of KindleHighlight objects."""
    # Remove BOM if present
    if content.startswith("\ufeff"):
        content = content[1:]

    # Delimiter is 10 equal signs
    raw_entries = re.split(r"(?m)^={10,}\s*$", content)
    highlights: List[KindleHighlight] = []
    seen_hashes: set[str] = set()

    for entry in raw_entries:
        entry_text = entry.strip()
        if not entry_text:
            continue

        lines = [line.strip() for line in entry_text.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        title_line = lines[0]
        meta_line = lines[1]
        text_lines = lines[2:]

        # If it's a bookmark without text, skip
        highlight_text = " ".join(text_lines).strip()
        if not highlight_text:
            continue

        raw_title, author = extract_title_and_author(title_line)
        clean_title = clean_filename(raw_title)
        h_type, location, page, timestamp = parse_metadata_line(meta_line)

        # Unique clipping ID to prevent identical duplicates within the source file
        hash_input = f"{clean_title}:{author}:{location}:{highlight_text}"
        clipping_id = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:12]

        if clipping_id in seen_hashes:
            continue
        seen_hashes.add(clipping_id)

        highlights.append(
            KindleHighlight(
                book_title=raw_title,
                clean_title=clean_title,
                author=author,
                location=location,
                page=page,
                timestamp=timestamp,
                text=highlight_text,
                highlight_type=h_type,
                clipping_id=clipping_id,
            )
        )

    return highlights


def parse_clippings_file(file_path: str) -> List[KindleHighlight]:
    """Read and parse a My Clippings.txt file from disk."""
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                content = f.read()
            return parse_clippings_content(content)
        except UnicodeDecodeError:
            continue

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return parse_clippings_content(content)


def group_highlights_by_book(highlights: List[KindleHighlight]) -> Dict[str, List[KindleHighlight]]:
    """Group highlights by sanitized book title."""
    grouped: Dict[str, List[KindleHighlight]] = {}
    for h in highlights:
        grouped.setdefault(h.clean_title, []).append(h)
    return grouped
