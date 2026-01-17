#!/usr/bin/python3

"""
Focus
---------------------------------

Features
- Displays one text file at a time from a configurable directory.
- Mouse wheel scrolls within the current record; hold Ctrl and wheel to load the previous/next page.
- Page jump entry (Ctrl+E) and gap-tolerant grep entry (Ctrl+F) stay in the header.
- Grep matches render in red and can show all matching pages in a single scrollable view.
- Ctrl+Shift+A opens the AI panel and focuses the RAG question box.
- Keyboard shortcuts: Up = previous, Down = next, Home/End = first/last.
- Scrollbars track your position while you browse.

Dependencies
- Python 3.10+
- PyGObject (gi), GTK 4, Libadwaita 1
  Ubuntu/Debian example: `sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1`

Run
- `python focus.py`

"""
from __future__ import annotations

import bisect
import io
import json
import os
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gdk, GObject, Gtk, Pango  # type: ignore

# =====================
# Configuration
# =====================
APPLICATION_ID = "com.mcglaw.Focus"
APPLICATION_NAME = "Focus"

GLib.set_application_name(APPLICATION_NAME)

CONFIG_FILE = Path(__file__).with_name("config.json")
CONFIG_KEY_INPUT_DIR = "input_dir"
CONFIG_KEY_REGEX_DIR = "regex_dir"
CONFIG_KEY_API_URL = "api_url"
CONFIG_KEY_MODEL_ID = "model_id"
CONFIG_KEY_API_KEY = "api_key"
CONFIG_KEY_PAGE_API_URL = "page_api_url"
CONFIG_KEY_PAGE_MODEL_ID = "page_model_id"
CONFIG_KEY_PAGE_API_KEY = "page_api_key"
CONFIG_KEY_RANGE_API_URL = "range_api_url"
CONFIG_KEY_RANGE_MODEL_ID = "range_model_id"
CONFIG_KEY_RANGE_API_KEY = "range_api_key"
CONFIG_KEY_SUMMARIZATION_PROMPT = "summarization_prompt"
CONFIG_KEY_PAGE_PROMPT = "page_summarization_prompt"
CONFIG_KEY_RANGE_PROMPT = "range_summarization_prompt"
CONFIG_KEY_VOYAGE_API_KEY = "voyage_api_key"
CONFIG_KEY_VOYAGE_MODEL = "voyage_model"
CONFIG_KEY_RAG_MODEL = "rag_model_id"
CONFIG_KEY_RAG_PROMPT = "rag_prompt"
CONFIG_KEY_RAG_API_URL = "rag_api_url"
CONFIG_KEY_RAG_API_KEY = "rag_api_key"
CONFIG_KEY_RAG_CHUNK_COUNT = "rag_chunk_count"
CONFIG_KEY_SUMMARY_FILE = "summary_file"
CONFIG_KEY_SUMMARY_READ_POSITIONS = "summary_read_positions"
CONFIG_KEY_FONT_SIZE_PT = "font_size_pt"
CONFIG_KEY_AI_FONT_SIZE_PT = "ai_font_size_pt"
CONFIG_KEY_HIGHLIGHT_PHRASES = "highlight_phrases"
DEFAULT_INPUT_DIR = Path.home().resolve(strict=False)
DEFAULT_REGEX_DIR = Path(__file__).with_name("regexes")
DEFAULT_SUMMARIZATION_PROMPT = (
    "Summarize the provided court transcript in 3–5 concise bullet points. "
    "Highlight the core issues, who is speaking, and any rulings or key facts. "
    "If the text is incomplete or appears truncated, mention that plainly."
)
DEFAULT_RAG_PROMPT = (
    'I will ask you a question about a child welfare case. Below are the case details and transcripts of hearings '
    'and reports. Based on this material, please answer the question while integrating direct quotes taken directly '
    'from hearings or reports. The direct quotes must be bold. Begin your answer with a heading named "Answer:" Always '
    'respond in English. Here is the material:'
)
DEFAULT_RAG_CHUNK_COUNT = 8

# =====================
# UI Defaults
# =====================
MAX_BREAKS = 2
DEFAULT_TEXT_COLOR = "alpha(@window_fg_color, 0.68)"
DEFAULT_FONT_SIZE_PT = 11
DEFAULT_AI_FONT_SIZE_PT = 12
DEFAULT_MATCH_COLOR = "#ee122a"         # red
DEFAULT_QUOTED_PHRASE_ALPHA = 1.0
DEFAULT_AI_PANEL_BG_COLOR = "alpha(@window_fg_color, 0.08)"
DEFAULT_HIGHLIGHT_COLOR = "#12aa2b"     # green
SIDEBAR_TREE_INDENT = 10
AI_OUTPUT_MIN_HEIGHT = 140
AI_OUTPUT_MAX_HEIGHT = 480
AI_OUTPUT_LINE_HEIGHT = 1.25
CONTINUOUS_PAGE_BATCH = 25
CONTINUOUS_SCROLL_THRESHOLD_PX = 800
AI_VIEW_SUMMARIZE = "summarize"
AI_VIEW_QA = "qa"
AI_VIEW_FILE = "show-file"
VIEW_ONE_ID = "view1"
VIEW_TWO_ID = "view2"
VIEW_LABELS = {
    VIEW_ONE_ID: "Primary View",
    VIEW_TWO_ID: "Secondary View",
}


def _read_config() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        return {}
    try:
        raw = CONFIG_FILE.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _write_config(config: dict[str, Any]) -> None:
    serializable: dict[str, Any] = {}
    for key, value in config.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, Path):
            serializable[key] = str(value)
        else:
            serializable[key] = value
    try:
        CONFIG_FILE.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    except OSError:
        pass


def _normalize_input_dir(path: Path) -> Path:
    if path.name in {"text_record", "images"}:
        parent = path.parent
        if str(parent):
            return parent
    return path


def load_input_dir_from_config() -> Path:
    config = _read_config()
    candidate = config.get(CONFIG_KEY_INPUT_DIR)
    if isinstance(candidate, str) and candidate.strip():
        resolved = Path(candidate).expanduser().resolve(strict=False)
        normalized = _normalize_input_dir(resolved)
        if normalized != resolved:
            config[CONFIG_KEY_INPUT_DIR] = str(normalized)
            _write_config(config)
        return normalized
    normalized_default = _normalize_input_dir(DEFAULT_INPUT_DIR)
    config[CONFIG_KEY_INPUT_DIR] = str(normalized_default)
    _write_config(config)
    return normalized_default


def save_input_dir_to_config(path: Path) -> None:
    config = _read_config()
    resolved = path.expanduser().resolve(strict=False)
    normalized = _normalize_input_dir(resolved)
    config[CONFIG_KEY_INPUT_DIR] = str(normalized)
    _write_config(config)


def _text_dir_from_root(root: Path) -> Path:
    if root.name == "text_record":
        return root
    return root / "text_record"


def _images_dir_from_root(root: Path) -> Path:
    base = root
    if base.name == "text_record":
        base = base.parent
    return base / "images"


def load_regex_dir_from_config() -> Path:
    config = _read_config()
    candidate = config.get(CONFIG_KEY_REGEX_DIR)
    if isinstance(candidate, str) and candidate.strip():
        return Path(candidate).expanduser().resolve(strict=False)
    config[CONFIG_KEY_REGEX_DIR] = str(DEFAULT_REGEX_DIR)
    _write_config(config)
    return DEFAULT_REGEX_DIR


def save_regex_dir_to_config(path: Path) -> None:
    config = _read_config()
    config[CONFIG_KEY_REGEX_DIR] = str(path)
    _write_config(config)


def _normalize_highlight_phrases(value: Any) -> list[str]:
    if isinstance(value, str):
        candidates: Iterable[str] = value.splitlines()
    elif isinstance(value, list):
        candidates = [item for item in value if isinstance(item, str)]
    else:
        candidates = []
    cleaned: list[str] = []
    seen: set[str] = set()
    for phrase in candidates:
        trimmed = phrase.strip()
        if not trimmed or trimmed in seen:
            continue
        seen.add(trimmed)
        cleaned.append(trimmed)
    return cleaned


def _format_highlight_phrases(phrases: Iterable[str]) -> str:
    return "\n".join(phrase for phrase in phrases if phrase.strip())


def load_summary_file_from_config(base_dir: Path | None = None) -> Path | None:
    config = _read_config()
    candidate = config.get(CONFIG_KEY_SUMMARY_FILE)
    if isinstance(candidate, str) and candidate.strip():
        raw_path = Path(candidate).expanduser()
        if not raw_path.is_absolute() and base_dir:
            raw_path = base_dir / raw_path
        return raw_path.resolve(strict=False)
    return None


def save_summary_file_to_config(path: Path) -> None:
    config = _read_config()
    config[CONFIG_KEY_SUMMARY_FILE] = str(path.expanduser().resolve(strict=False))
    _write_config(config)


def load_summary_read_positions() -> dict[str, float]:
    config = _read_config()
    raw = config.get(CONFIG_KEY_SUMMARY_READ_POSITIONS)
    if not isinstance(raw, dict):
        return {}
    positions: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        try:
            fraction = float(value)
        except (TypeError, ValueError):
            continue
        positions[key] = min(1.0, max(0.0, fraction))
    return positions


def save_summary_read_positions(positions: dict[str, float]) -> None:
    config = _read_config()
    config[CONFIG_KEY_SUMMARY_READ_POSITIONS] = positions
    _write_config(config)


def _coerce_font_size(value: Any, default: int) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError):
        return default
    return min(48, max(8, size))


def _coerce_rag_chunk_count(value: Any, default: int) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return default
    return min(50, max(1, count))


def load_font_preferences() -> tuple[int, int]:
    config = _read_config()
    base = _coerce_font_size(config.get(CONFIG_KEY_FONT_SIZE_PT), DEFAULT_FONT_SIZE_PT)
    ai_default = max(base, DEFAULT_AI_FONT_SIZE_PT)
    ai = _coerce_font_size(config.get(CONFIG_KEY_AI_FONT_SIZE_PT), ai_default)
    return base, ai


def save_font_preferences(font_size_pt: int, ai_font_size_pt: int) -> None:
    config = _read_config()
    config[CONFIG_KEY_FONT_SIZE_PT] = int(font_size_pt)
    config[CONFIG_KEY_AI_FONT_SIZE_PT] = int(ai_font_size_pt)
    _write_config(config)


LONGFORM_DATE_RE = re.compile(
    r"\b("
    r"January|February|March|April|May|June|July|August|September|October|November|December"
    r"|JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER"
    r")\s*([0-9]{1,2})\s*,\s*([0-9]{4})\b"
)
NUMERIC_DATE_RE = re.compile(r"\b([0-1]?\d)/([0-3]?\d)/([0-9]{4})\b")
MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _normalize_longform_spacing(line: str) -> str:
    def _fix(match: re.Match[str]) -> str:
        month = match.group(1)
        day = match.group(2)
        year = match.group(3)
        return f"{month} {int(day)}, {year}"

    return LONGFORM_DATE_RE.sub(_fix, line)


def _extract_date_key(line: str) -> str | None:
    match = LONGFORM_DATE_RE.search(line)
    if match:
        month_txt, day, year = match.group(1), int(match.group(2)), int(match.group(3))
        month_num = MONTHS.get(month_txt.lower())
        if month_num:
            return f"{year:04d}-{month_num:02d}-{day:02d}"

    numeric_match = NUMERIC_DATE_RE.search(line)
    if numeric_match:
        month, day, year = (
            int(numeric_match.group(1)),
            int(numeric_match.group(2)),
            int(numeric_match.group(3)),
        )
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def _looks_like_heading(line: str) -> bool:
    if not line or line[:1].isspace():
        return False
    return re.fullmatch(r"[A-Z][A-Z ]*(?:\s+\d+)?", line.strip()) is not None


def _ensure_one_blank_before_headings(lines: list[str]) -> list[str]:
    out: list[str] = []
    seen_any_content = False
    for ln in lines:
        if _looks_like_heading(ln):
            if seen_any_content:
                if len(out) == 0 or out[-1] != "":
                    out.append("")
                while len(out) >= 2 and out[-1] == "" and out[-2] == "":
                    out.pop(-2)
            out.append(ln)
            seen_any_content = True
        else:
            out.append(ln)
            if ln.strip():
                seen_any_content = True
    while out and out[0] == "":
        out.pop(0)
    return out


def clean_toc_text(original_text: str) -> tuple[str, int, int]:
    had_trailing_newline = original_text.endswith("\n")
    lines = original_text.splitlines()

    lines = [_normalize_longform_spacing(ln) for ln in lines]

    seen: set[str] = set()
    dedup_exact: list[str] = []
    removed_exact = 0
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            dedup_exact.append(ln)
        else:
            removed_exact += 1

    in_minutes = False
    dates_seen_in_minutes: set[str] = set()
    body: list[str] = []
    removed_minutes_dupes = 0
    for ln in dedup_exact:
        if _looks_like_heading(ln):
            in_minutes = ln.strip().startswith("MINUTES")
            dates_seen_in_minutes.clear()
            body.append(ln)
            continue

        if in_minutes:
            key = _extract_date_key(ln)
            if key is not None:
                if key in dates_seen_in_minutes:
                    removed_minutes_dupes += 1
                    continue
                dates_seen_in_minutes.add(key)

        body.append(ln)

    final_lines = _ensure_one_blank_before_headings(body)
    cleaned = "\n".join(final_lines) + ("\n" if had_trailing_newline else "")
    return cleaned, removed_exact, removed_minutes_dupes


@dataclass
class AiSettings:
    api_url: str
    model_id: str
    api_key: str
    page_api_url: str
    page_model_id: str
    page_api_key: str
    range_api_url: str
    range_model_id: str
    range_api_key: str
    page_prompt: str
    range_prompt: str
    voyage_api_key: str
    voyage_model: str
    rag_llm_model: str
    rag_prompt: str
    rag_api_url: str
    rag_api_key: str
    rag_chunk_count: int
    highlight_phrases: list[str]

    def page_credentials(self) -> tuple[str, str, str]:
        return (
            self.page_api_url.strip() or self.api_url.strip(),
            self.page_model_id.strip() or self.model_id.strip(),
            self.page_api_key.strip() or self.api_key.strip(),
        )

    def range_credentials(self) -> tuple[str, str, str]:
        return (
            self.range_api_url.strip() or self.api_url.strip(),
            self.range_model_id.strip() or self.model_id.strip(),
            self.range_api_key.strip() or self.api_key.strip(),
        )

    def rag_credentials(self) -> tuple[str, str]:
        page_api_url, _, page_api_key = self.page_credentials()
        return (
            self.rag_api_url.strip() or page_api_url,
            self.rag_api_key.strip() or page_api_key,
        )

    def is_configured(self) -> bool:
        page_api_url, page_model_id, page_api_key = self.page_credentials()
        range_api_url, range_model_id, range_api_key = self.range_credentials()
        return all(
            value.strip()
            for value in (
                page_api_url,
                page_model_id,
                page_api_key,
                range_api_url,
                range_model_id,
                range_api_key,
                self.page_prompt,
                self.range_prompt,
            )
        )

    def is_rag_ready(self) -> bool:
        rag_api_url, rag_api_key = self.rag_credentials()
        return all(
            value.strip()
            for value in (
                self.voyage_api_key,
                self.voyage_model,
                self.rag_llm_model,
                self.rag_prompt,
                rag_api_url,
                rag_api_key,
            )
        )


@dataclass
class AiOutputView:
    raw: str = ""
    view: Gtk.TextView | None = None
    buffer: Gtk.TextBuffer | None = None
    scroller: Gtk.ScrolledWindow | None = None
    link_tags: list[Gtk.TextTag] = field(default_factory=list)
    link_lookup: dict[Gtk.TextTag, str] = field(default_factory=dict)
    motion_controller: Gtk.EventControllerMotion | None = None
    click_gesture: Gtk.GestureClick | None = None
    focus_controller: Gtk.EventControllerFocus | None = None


@dataclass
class FocusViewState:
    name: str
    current_index: int = 0
    show_image: bool = False
    sidebar_visible: bool = True
    ai_panel_visible: bool = False
    continuous_view: bool = False
    continuous_text: str | None = None
    continuous_pages_order: list[int] = field(default_factory=list)
    continuous_loaded_count: int = 0
    continuous_loading: bool = False
    grep_phrase_raw: str | None = None
    grep_regex: re.Pattern[str] | None = None
    grep_hits: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    matching_pages: list[int] = field(default_factory=list)
    matching_lookup: dict[int, int] = field(default_factory=dict)
    grep_combined_text: str | None = None
    grep_combined_highlights: list[tuple[int, int]] = field(default_factory=list)
    showing_grep_results: bool = False
    ai_active_view: str = AI_VIEW_QA
    ai_output_raw: dict[str, str] = field(
        default_factory=lambda: {AI_VIEW_SUMMARIZE: "", AI_VIEW_QA: ""}
    )
    ai_status_text: str = ""
    ai_spinning: bool = False
    ai_request_generation: int = 0
    ai_in_flight: bool = False
    ai_cancel_event: threading.Event | None = None
    ai_stream_thread: threading.Thread | None = None
    ai_range_text: str = ""
    rag_question_text: str = ""
    sidebar_expanded: list[str] = field(default_factory=list)

def load_ai_settings() -> AiSettings:
    config = _read_config()
    api_url = str(config.get(CONFIG_KEY_API_URL, "") or "").strip()
    model_id = str(config.get(CONFIG_KEY_MODEL_ID, "") or "").strip()
    api_key = str(config.get(CONFIG_KEY_API_KEY, "") or "").strip()
    page_api_url = str(config.get(CONFIG_KEY_PAGE_API_URL, "") or "").strip()
    page_model_id = str(config.get(CONFIG_KEY_PAGE_MODEL_ID, "") or "").strip()
    page_api_key = str(config.get(CONFIG_KEY_PAGE_API_KEY, "") or "").strip()
    range_api_url = str(config.get(CONFIG_KEY_RANGE_API_URL, "") or "").strip()
    range_model_id = str(config.get(CONFIG_KEY_RANGE_MODEL_ID, "") or "").strip()
    range_api_key = str(config.get(CONFIG_KEY_RANGE_API_KEY, "") or "").strip()
    fallback_prompt = str(config.get(CONFIG_KEY_SUMMARIZATION_PROMPT, DEFAULT_SUMMARIZATION_PROMPT) or "").strip()
    page_prompt = str(config.get(CONFIG_KEY_PAGE_PROMPT, fallback_prompt) or fallback_prompt).strip()
    range_prompt = str(config.get(CONFIG_KEY_RANGE_PROMPT, fallback_prompt) or fallback_prompt).strip()
    rag_prompt = str(config.get(CONFIG_KEY_RAG_PROMPT, DEFAULT_RAG_PROMPT) or DEFAULT_RAG_PROMPT).strip()
    voyage_model = str(config.get(CONFIG_KEY_VOYAGE_MODEL, "voyage-law-2") or "voyage-law-2").strip()
    rag_api_url = str(config.get(CONFIG_KEY_RAG_API_URL, "") or "").strip()
    rag_api_key = str(config.get(CONFIG_KEY_RAG_API_KEY, "") or "").strip()
    rag_chunk_count = _coerce_rag_chunk_count(
        config.get(CONFIG_KEY_RAG_CHUNK_COUNT),
        DEFAULT_RAG_CHUNK_COUNT,
    )
    highlight_phrases = _normalize_highlight_phrases(config.get(CONFIG_KEY_HIGHLIGHT_PHRASES))
    return AiSettings(
        api_url=api_url,
        model_id=model_id,
        api_key=api_key,
        page_api_url=page_api_url,
        page_model_id=page_model_id,
        page_api_key=page_api_key,
        range_api_url=range_api_url,
        range_model_id=range_model_id,
        range_api_key=range_api_key,
        page_prompt=page_prompt or DEFAULT_SUMMARIZATION_PROMPT,
        range_prompt=range_prompt or DEFAULT_SUMMARIZATION_PROMPT,
        voyage_api_key=str(config.get(CONFIG_KEY_VOYAGE_API_KEY, "") or "").strip(),
        voyage_model=voyage_model or "voyage-law-2",
        rag_llm_model=str(config.get(CONFIG_KEY_RAG_MODEL, "") or "").strip(),
        rag_prompt=rag_prompt or DEFAULT_RAG_PROMPT,
        rag_api_url=rag_api_url,
        rag_api_key=rag_api_key,
        rag_chunk_count=rag_chunk_count,
        highlight_phrases=highlight_phrases,
    )


def save_ai_settings(settings: AiSettings) -> None:
    config = _read_config()
    config[CONFIG_KEY_API_URL] = settings.api_url
    config[CONFIG_KEY_MODEL_ID] = settings.model_id
    config[CONFIG_KEY_API_KEY] = settings.api_key
    config[CONFIG_KEY_PAGE_API_URL] = settings.page_api_url
    config[CONFIG_KEY_PAGE_MODEL_ID] = settings.page_model_id
    config[CONFIG_KEY_PAGE_API_KEY] = settings.page_api_key
    config[CONFIG_KEY_RANGE_API_URL] = settings.range_api_url
    config[CONFIG_KEY_RANGE_MODEL_ID] = settings.range_model_id
    config[CONFIG_KEY_RANGE_API_KEY] = settings.range_api_key
    config[CONFIG_KEY_SUMMARIZATION_PROMPT] = settings.page_prompt or DEFAULT_SUMMARIZATION_PROMPT
    config[CONFIG_KEY_PAGE_PROMPT] = settings.page_prompt or DEFAULT_SUMMARIZATION_PROMPT
    config[CONFIG_KEY_RANGE_PROMPT] = settings.range_prompt or DEFAULT_SUMMARIZATION_PROMPT
    config[CONFIG_KEY_VOYAGE_API_KEY] = settings.voyage_api_key
    config[CONFIG_KEY_VOYAGE_MODEL] = settings.voyage_model or "voyage-law-2"
    config[CONFIG_KEY_RAG_MODEL] = settings.rag_llm_model
    config[CONFIG_KEY_RAG_PROMPT] = settings.rag_prompt or DEFAULT_RAG_PROMPT
    config[CONFIG_KEY_RAG_API_URL] = settings.rag_api_url
    config[CONFIG_KEY_RAG_API_KEY] = settings.rag_api_key
    config[CONFIG_KEY_RAG_CHUNK_COUNT] = _coerce_rag_chunk_count(
        settings.rag_chunk_count,
        DEFAULT_RAG_CHUNK_COUNT,
    )
    config[CONFIG_KEY_HIGHLIGHT_PHRASES] = settings.highlight_phrases
    _write_config(config)

_FLAG_MAP = {
    "i": re.IGNORECASE,
    "m": re.MULTILINE,
    "s": re.DOTALL,
    "x": re.VERBOSE,
}

CONTINUOUS_ICON_ON_CHOICES = (
    "view-continuous-symbolic",
    "view-list-symbolic",
    "view-grid-symbolic",
)
CONTINUOUS_ICON_OFF_CHOICES = (
    "zoom-original-symbolic",
    "view-restore-symbolic",
    "view-dual-symbolic",
)
IMAGE_ICON_ON_CHOICES = (
    "image-x-generic-symbolic",
    "insert-image-symbolic",
    "image-x-generic",
)
IMAGE_ICON_OFF_CHOICES = (
    "font-size-symbolic",
    "image-x-generic-symbolic",
    "image-missing",
)
AI_PANEL_ICON_CHOICES = (
    "computer-chip-symbolic",
    "panel-top-symbolic",
    "view-dual-symbolic",
    "window-restore-symbolic",
)


def _parse_flag_line(line: str) -> int | None:
    stripped = line.strip()
    if not stripped.lower().startswith("# flags:"):
        return None
    parts = stripped.split(":", 1)
    if len(parts) != 2:
        return None
    flags = 0
    for token in parts[1].replace(",", " ").split():
        flags |= _FLAG_MAP.get(token.lower(), 0)
    return flags or None


def _load_regex_patterns(regex_dir: Path) -> dict[str, list[tuple[re.Pattern[str], bool]]]:
    patterns: dict[str, list[tuple[re.Pattern[str], bool]]] = {}
    try:
        candidates = [
            p
            for p in regex_dir.iterdir()
            if p.is_file() and not p.name.startswith(".") and p.name != "_order.txt"
        ]
    except OSError as exc:  # noqa: BLE001
        raise FileNotFoundError(f"Unable to read regex directory: {regex_dir}") from exc

    for path in sorted(candidates, key=lambda p: p.name.lower()):
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError as exc:  # noqa: BLE001
            raise FileNotFoundError(f"Unable to read regex file: {path}") from exc

        entries: list[tuple[re.Pattern[str], bool]] = []
        current_flags = re.MULTILINE
        for line in raw:
            maybe_flags = _parse_flag_line(line)
            if maybe_flags is not None:
                current_flags = maybe_flags
                continue
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                continue
            try:
                compiled = re.compile(line, current_flags)
            except re.error as exc:
                raise ValueError(f"{path.name}: invalid regex '{line}': {exc}") from exc
            entries.append((compiled, compiled.groups > 0))

        if entries:
            patterns[path.stem] = entries

    return patterns


def generate_toc_text(
    text_dir: Path,
    regex_dir: Path,
    *,
    update_progress: Callable[[float], None] | None = None,
) -> str:
    if not text_dir.exists() or not text_dir.is_dir():
        raise FileNotFoundError(f"Text directory not found: {text_dir}")
    if not regex_dir.exists() or not regex_dir.is_dir():
        raise FileNotFoundError(f"Regex directory not found: {regex_dir}")

    pages: list[tuple[int, Path]] = []
    for entry in text_dir.iterdir():
        if not entry.is_file():
            continue
        match = PAGE_RE.match(entry.name)
        if not match:
            continue
        page_num = int(match.group("num"))
        pages.append((page_num, entry))
    pages.sort(key=lambda item: item[0])
    if not pages:
        raise FileNotFoundError(f"No page text files found in: {text_dir}")

    categories = _load_regex_patterns(regex_dir)
    if not categories:
        raise FileNotFoundError(f"No regex files found in: {regex_dir}")

    results: dict[str, list[tuple[str, int]]] = {cat: [] for cat in categories}
    seen: dict[str, set[tuple[str, int]]] = {cat: set() for cat in categories}

    total = len(pages)
    for idx, (page_num, path) in enumerate(pages, start=1):
        try:
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            raw_text = ""
        normalized = normalize_text_for_search(raw_text)
        for cat, pattern_entries in categories.items():
            for compiled, has_groups in pattern_entries:
                try:
                    for match in compiled.finditer(normalized):
                        value = match.group(1) if has_groups and match.lastindex else match.group(0)
                        value = value.strip()
                        if not value:
                            continue
                        key = (value, page_num)
                        if key in seen[cat]:
                            continue
                        results[cat].append((value, page_num))
                        seen[cat].add(key)
                except re.error:
                    continue
        if update_progress:
            update_progress(idx / max(total, 1))

    first_page_fallback = pages[0][0]
    out_lines: list[str] = []
    for cat in sorted(categories, key=lambda s: s.lower()):
        cat_hits = sorted(results.get(cat, []), key=lambda t: t[1])
        first_page = cat_hits[0][1] if cat_hits else first_page_fallback
        out_lines.append(f"{cat} {first_page:04d}")
        for title, pg in cat_hits:
            out_lines.append(f"\t{title} {pg:04d}")
        out_lines.append("")

    return "\n".join(out_lines) + "\n"


@dataclass
class TocBookmark:
    title: str
    page: int


@dataclass
class TocCategory:
    title: str
    page: int | None
    bookmarks: list[TocBookmark]


class FocusSidebarItem(GObject.GObject):
    __gtype_name__ = "FocusSidebarItem"

    def __init__(
        self,
        title: str,
        page: int | None,
        *,
        kind: str,
        children: list["FocusSidebarItem"] | None = None,
    ) -> None:
        super().__init__()
        self.title = title
        self.page = page
        self.kind = kind
        self._children_store: Gio.ListStore | None = None
        if children:
            store = Gio.ListStore(item_type=FocusSidebarItem)
            for child in children:
                store.append(child)
            self._children_store = store

    def get_children_model(self) -> Gio.ListModel | None:
        return self._children_store

    @classmethod
    def from_category(cls, category: TocCategory) -> "FocusSidebarItem":
        children = [
            cls(title=bookmark.title, page=bookmark.page, kind="bookmark")
            for bookmark in category.bookmarks
        ]
        return cls(title=category.title, page=category.page, kind="category", children=children or None)

APP_CHROME_CSS = (
    """
window.background.focus-window {
  background: @view_bg_color;
}

navigation-split-view.focus-split,
navigation-split-view.focus-split navigation-sidebar,
navigation-split-view.focus-split navigation-sidebar > stack {
  background: @view_bg_color;
}

box.focus-sidebar {
  background: @view_bg_color;
}

listview.focus-sidebar-listview,
listview.focus-sidebar-listview row,
listbox.focus-sidebar-listview,
listbox.focus-sidebar-listview row {
  background: transparent;
}

.focus-sidebar-listbox-row,
.focus-sidebar-listbox-row:hover,
.focus-sidebar-listbox-row:selected,
.focus-sidebar-listbox-row:focus {
  background: transparent;
  box-shadow: none;
  padding: 0;
}

.focus-sidebar-row {
  min-height: 0px;
  transition: background-color 120ms ease;
  border-radius: 12px;
  padding: 0;
}

.focus-sidebar-row.focus-sidebar-category {
  background-color: transparent;
}

.focus-sidebar-row.focus-sidebar-category:hover,
.focus-sidebar-row.focus-sidebar-category.focus-sidebar-category-expanded {
  background-color: transparent;
}

.focus-sidebar-row.focus-sidebar-category .title,
.focus-sidebar-row.focus-sidebar-category label,
.focus-sidebar-row.focus-sidebar-bookmark .title,
.focus-sidebar-row.focus-sidebar-bookmark label,
.focus-sidebar-expand-button {
  color: alpha(@window_fg_color, 0.62);
}

.focus-sidebar-row.focus-sidebar-bookmark {
  background-color: transparent;
  padding-top: 4px;
  padding-bottom: 4px;
  margin-top: 2px;
  margin-bottom: 2px;
}

.focus-sidebar-row.focus-sidebar-bookmark:hover {
  background-color: alpha(@window_fg_color, 0.04);
}

.focus-sidebar-row.focus-sidebar-bookmark-active,
.focus-sidebar-row.focus-sidebar-bookmark-active:hover {
  background-color: alpha(@window_fg_color, 0.08);
}

.focus-sidebar-expand-button {
  min-height: 28px;
  min-width: 28px;
  padding: 0;
  margin-right: -4px;
  border-radius: 999px;
  background: transparent;
  box-shadow: none;
}

.focus-sidebar-row .title {
  margin-left: -2px;
}

.focus-sidebar-row.focus-sidebar-category {
  padding-top: 0px;
  padding-bottom: 0px;
}

.focus-sidebar-expand-button:hover,
.focus-sidebar-expand-button:checked,
.focus-sidebar-expand-button:active {
  background: transparent;
  box-shadow: none;
}

.focus-root-scroller,
.focus-root-scroller > viewport {
  background: @view_bg_color;
}

/* deactivate for focus minutes */
/*headerbar.flat.focus-header {
  background: #2e7d32;
  color: #ffffff;
}*/

.focus-scroller,
.focus-scroller > viewport {
  background: @view_bg_color;
}

.focus-scroller > viewport {
  padding-top: 10px;
}

.ai-output-frame {
  background-color: __AI_PANEL_BG__;
  border-radius: 16px;
  padding: 10px;
}

.ai-output-view {
  background: transparent;
}

.no-bold {
  font-weight: normal;
}

.focus-toggle-icon {
  color: alpha(@window_fg_color, 0.62);
}

.focus-view-toggle:not(:checked),
.focus-view-toggle:not(:checked) label {
  color: alpha(@window_fg_color, 0.62);
}

#page-text {
  background-color: alpha(@window_fg_color, 0.08);
  border-top-left-radius: 16px;
}
"""
).replace("__AI_PANEL_BG__", DEFAULT_AI_PANEL_BG_COLOR)
_chrome_provider = Gtk.CssProvider()
_chrome_provider.load_from_data(APP_CHROME_CSS.encode("utf-8"))
_display = Gdk.Display.get_default()
if _display:
    Gtk.StyleContext.add_provider_for_display(
        _display,
        _chrome_provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
    )


TOC_LINE_RE = re.compile(r"^(?P<title>.*\S)\s+(?P<page>\d+)\s*$")


def parse_toc_text(text: str) -> list[TocCategory]:
    categories: list[TocCategory] = []
    current: TocCategory | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        line = raw_line.rstrip()
        indent = len(line) - len(line.lstrip(" \t"))
        match = TOC_LINE_RE.match(line.strip())
        if not match:
            continue
        title = match.group("title").strip()
        try:
            page = int(match.group("page"))
        except ValueError:
            continue
        if indent > 0:
            if current is None:
                continue
            bookmark = TocBookmark(title=title, page=page)
            current.bookmarks.append(bookmark)
        else:
            current = TocCategory(title=title, page=page, bookmarks=[])
            categories.append(current)
    return categories


def read_toc_text(toc_path: Path) -> tuple[str, str | None]:
    try:
        text = toc_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "", None
    except OSError as exc:  # noqa: BLE001
        return "", f"Failed to read {toc_path.name}: {exc}"
    return text, None


NORMALIZE_REPLACEMENTS = {
    "\u2019": "'",
    "\u2018": "'",
    "\u201C": '"',
    "\u201D": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00A0": " ",
    "\u00B7": " ",
    "\u2022": " ",
}


def normalize_quotes_dashes(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for src, dst in NORMALIZE_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text


def normalize_text_for_search(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalize_quotes_dashes(text)


def normalize_text_for_search_with_map(text: str) -> tuple[str, list[int]]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_chars: list[str] = []
    norm_to_orig: list[int] = []
    for idx, ch in enumerate(text):
        normalized = unicodedata.normalize("NFKC", ch)
        if not normalized:
            continue
        for out_ch in normalized:
            replacement = NORMALIZE_REPLACEMENTS.get(out_ch, out_ch)
            for repl_ch in replacement:
                normalized_chars.append(repl_ch)
                norm_to_orig.append(idx)
    return "".join(normalized_chars), norm_to_orig


def preprocess_phrase(phrase: str) -> str:
    return normalize_quotes_dashes(phrase)


def build_word_pattern(word: str) -> str:
    parts: list[str] = []
    length = len(word)
    for i, ch in enumerate(word):
        if ch == "-":
            parts.append(r"\-(?:[ ]*\n[ ]*)?")
        elif ch in ("'", '"'):
            parts.append(r"['\"]")
        else:
            parts.append(re.escape(ch))
        if ch.isalnum() and i + 1 < length and word[i + 1].isalnum():
            parts.append(r"(?:[ ]+)?")
    return "".join(parts)


def build_pattern(phrase: str, max_breaks: int = MAX_BREAKS) -> str:
    words = [w for w in re.split(r"\s+", phrase) if w]
    words = [build_word_pattern(w) for w in words]

    newline_alts = []
    for count in range(1, max_breaks + 1):
        newline_alts.append(r"(?:[ \t]*\n)" * count + r"[ \t]*")

    alts = [r"(?:[ \t]+)"] + newline_alts
    sep = r"(?:%s)(?:\d+[ \t]*)*" % ("|".join(alts) if alts else r"(?:[ \t]+)")
    return r"(?x)" + sep.join(words)


PAGE_RE = re.compile(r"^(?P<num>\d{4})\.txt$")
PAGE_HEADER_LINE_RE = re.compile(r"^(?P<num>\d{4})(?P<rest>[^\n]*)\n\n", re.MULTILINE)
AI_LINK_SPAN_RE = re.compile(r'(?:\"|“)(.+?)(?:\"|”)|\*\*(.+?)\*\*', re.DOTALL)
LINK_TRAILING_PUNCTUATION = ",.;:!?)]"


def split_link_phrase(phrase: str) -> tuple[str, str]:
    """Split a phrase into linkable text and trailing punctuation."""
    end = len(phrase)
    while end > 0 and phrase[end - 1] in LINK_TRAILING_PUNCTUATION:
        end -= 1
    core = phrase[:end].rstrip()
    trailing = phrase[end:]
    return core, trailing


class Focus(Adw.Application):
    def __init__(self, *, input_override: Path | None = None) -> None:
        super().__init__(
            application_id=APPLICATION_ID,
            flags=Gio.ApplicationFlags.FLAGS_NONE,
        )
        try:
            style_manager = Adw.StyleManager.get_default()
            style_manager.set_color_scheme(Adw.ColorScheme.DEFAULT)
            style_manager.connect("notify::color-scheme", self._on_color_scheme_changed)
        except Exception:
            pass
        self.connect("activate", self.on_activate)

        if input_override is not None:
            self.input_dir = input_override
        else:
            self.input_dir = load_input_dir_from_config()
        self.regex_dir: Path = load_regex_dir_from_config()
        self._font_size_pt, self._ai_font_size_pt = load_font_preferences()

        self.pages: list[int] = []
        self.page_to_path: dict[int, Path] = {}
        self.current_index: int = 0

        self.win: Adw.ApplicationWindow | None = None
        self._input_dir_dialog: Gtk.FileDialog | None = None
        self._regex_dir_dialog: Gtk.FileDialog | None = None
        self._summary_file_dialog: Gtk.FileDialog | None = None
        self._summary_file_path: Path | None = load_summary_file_from_config(self.input_dir)
        self.textview: Gtk.TextView | None = None
        self.scroller: Gtk.ScrolledWindow | None = None
        self._grep_entry: Gtk.Entry | None = None
        self._center_label: Gtk.Label | None = None
        self._toc_window: TocWindow | None = None
        self._split_view: Adw.NavigationSplitView | None = None
        self._toc_sidebar_revealer: Gtk.Revealer | None = None
        self._toc_sidebar_overlay: Gtk.Overlay | None = None
        self._toc_sidebar_scroller: Gtk.ScrolledWindow | None = None
        self._toc_list_view: Gtk.ListBox | None = None
        self._toc_sidebar_root_store: Gio.ListStore | None = None
        self._toc_sidebar_tree_model: Gtk.TreeListModel | None = None
        self._toc_sidebar_button: Gtk.ToggleButton | None = None
        self._toc_sidebar_action: Gio.SimpleAction | None = None
        self._toc_sidebar_icon: Gtk.Image | None = None
        self._split_content_page: Adw.NavigationPage | None = None
        self._split_sidebar_page: Adw.NavigationPage | None = None
        self._toc_placeholder: Gtk.Widget | None = None
        self._toc_sidebar_has_items = False
        self._toc_sidebar_visible = True
        self._sidebar_button_guard = False
        self._continuous_view = False
        self._continuous_text: str | None = None
        self._continuous_action: Gio.SimpleAction | None = None
        self._continuous_pages_order: list[int] = []
        self._continuous_loaded_count = 0
        self._continuous_vadj_handler: int | None = None
        self._continuous_loading = False
        self._current_text_color = DEFAULT_TEXT_COLOR

        self._color_provider = Gtk.CssProvider()
        self._css_provider_registered = False

        self._page_back_ten_button: Gtk.Button | None = None
        self._page_back_one_button: Gtk.Button | None = None
        self._page_forward_one_button: Gtk.Button | None = None
        self._page_forward_ten_button: Gtk.Button | None = None
        self._page_status_label: Gtk.Label | None = None

        self._grep_phrase_raw: str | None = None
        self._grep_regex: re.Pattern[str] | None = None
        self._grep_hits: dict[int, list[tuple[int, int]]] = {}
        self._matching_pages: list[int] = []
        self._matching_lookup: dict[int, int] = {}
        self._grep_combined_text: str | None = None
        self._grep_combined_highlights: list[tuple[int, int]] = []
        self._showing_grep_results = False

        self._page_cache: dict[int, str] = {}
        self._page_search_cache: dict[int, str] = {}
        self._page_search_map_cache: dict[int, list[int]] = {}
        self._link_tags: list[Gtk.TextTag] = []
        self._link_tag_lookup: dict[Gtk.TextTag, tuple[str, str]] = {}
        self._ai_outputs: dict[str, AiOutputView] = {
            AI_VIEW_SUMMARIZE: AiOutputView(),
            AI_VIEW_QA: AiOutputView(),
        }
        self._ai_active_view = AI_VIEW_QA
        self._textview_click_gesture: Gtk.GestureClick | None = None
        self._textview_focus_controller: Gtk.EventControllerFocus | None = None
        self._textview_motion_controller: Gtk.EventControllerMotion | None = None
        self._summary_link_tags: list[Gtk.TextTag] = []
        self._summary_link_tag_lookup: dict[Gtk.TextTag, str] = {}
        self._summary_click_gesture: Gtk.GestureClick | None = None
        self._summary_focus_controller: Gtk.EventControllerFocus | None = None
        self._summary_motion_controller: Gtk.EventControllerMotion | None = None
        self._summary_search_entry: Gtk.SearchEntry | None = None
        self._summary_search_query = ""
        self._summary_search_matches: list[tuple[int, int]] = []
        self._summary_search_index = -1
        self._summary_search_tag: Gtk.TextTag | None = None
        self._summary_search_current_tag: Gtk.TextTag | None = None
        self._summary_view: Gtk.TextView | None = None
        self._summary_buffer: Gtk.TextBuffer | None = None
        self._summary_scroller: Gtk.ScrolledWindow | None = None
        self._summary_loaded_path: Path | None = None
        self._summary_raw = ""
        self._auto_loading_summary = False
        self._summary_scroll_positions = load_summary_read_positions()
        self._summary_scroll_save_source_id: int | None = None
        self._summary_scroll_handler_id: int | None = None
        self._summary_scroll_restore_guard = False
        self._edge_flash_source_id: int | None = None
        self._content_stack: Gtk.Stack | None = None
        self._image_scroller: Gtk.ScrolledWindow | None = None
        self._image_picture: Gtk.Picture | None = None
        self._show_image = False
        self._show_image_action: Gio.SimpleAction | None = None
        self._continuous_toggle_button: Gtk.ToggleButton | None = None
        self._continuous_icon: Gtk.Image | None = None
        self._continuous_button_guard = False
        self._continuous_icon_name_on = CONTINUOUS_ICON_ON_CHOICES[0]
        self._continuous_icon_name_off = self._continuous_icon_name_on
        self._show_image_button: Gtk.ToggleButton | None = None
        self._show_image_icon: Gtk.Image | None = None
        self._show_image_button_guard = False
        self._image_icon_name_on = IMAGE_ICON_ON_CHOICES[0]
        self._image_icon_name_off = self._image_icon_name_on
        self._toc_categories: list[TocCategory] = []
        self._toc_load_generation = 0
        self._ai_panel_revealer: Gtk.Revealer | None = None
        self._ai_panel_icon: Gtk.Image | None = None
        self._ai_view_stack: Adw.ViewStack | None = None
        self._ai_view_toggles: dict[str, Gtk.ToggleButton] = {}
        self._ai_view_toggle_guard = False
        self._ai_controls_stack: Gtk.Stack | None = None
        self._ai_status_label: Gtk.Label | None = None
        self._ai_spinner: Gtk.Spinner | None = None
        self._ai_range_entry: Gtk.Entry | None = None
        self._ai_panel_toggle: Gtk.ToggleButton | None = None
        self._choose_summary_button: Gtk.Button | None = None
        self._ai_stream_thread: threading.Thread | None = None
        self._ai_cancel_event: threading.Event | None = None
        self._ai_settings_window: AiSettingsWindow | None = None
        self._ai_settings: AiSettings = load_ai_settings()
        self._ai_in_flight = False
        self._ai_request_generation = 0
        self._rag_vectorstore: Any | None = None
        self._rag_case_details: str | None = None
        self._rag_load_thread: threading.Thread | None = None
        self._rag_load_generation = 0
        self._rag_load_error: str | None = None
        self._rag_loading = False
        self._rag_lock = threading.Lock()
        self._rag_question_entry: Gtk.Entry | None = None
        self._ai_panel_icon_name = AI_PANEL_ICON_CHOICES[0]
        self._views: dict[str, FocusViewState] = {
            VIEW_ONE_ID: FocusViewState(name=VIEW_LABELS[VIEW_ONE_ID]),
            VIEW_TWO_ID: FocusViewState(name=VIEW_LABELS[VIEW_TWO_ID]),
        }
        self._active_view_id = VIEW_ONE_ID
        self._view_buttons: dict[str, Gtk.ToggleButton] = {}
        self._view_button_guard = False

    @property
    def text_dir(self) -> Path:
        return _text_dir_from_root(self.input_dir)

    @property
    def images_dir(self) -> Path:
        return _images_dir_from_root(self.input_dir)

    def _choose_icon(self, *names: str) -> str:
        if not names:
            return ""
        display = self.win.get_display() if self.win else Gdk.Display.get_default()
        theme = Gtk.IconTheme.get_for_display(display) if display else None
        if theme:
            for name in names:
                try:
                    if theme.has_icon(name):
                        return name
                except TypeError:
                    continue
        return names[0]

    def _scan_pages(self) -> None:
        self.page_to_path.clear()
        self.pages.clear()
        self._page_cache.clear()
        self._page_search_cache.clear()
        self._page_search_map_cache.clear()
        text_dir = self.text_dir
        if not text_dir.exists():
            return
        for p in text_dir.iterdir():
            if not p.is_file():
                continue
            m = PAGE_RE.match(p.name)
            if m:
                num = int(m.group("num"))
                self.page_to_path[num] = p
        self.pages = sorted(self.page_to_path.keys())

    def _current_toc_path(self) -> Path:
        return self.text_dir / "toc.txt"

    def _load_toc_from_disk_async(self) -> None:
        toc_path = self._current_toc_path()
        self._toc_load_generation += 1
        generation = self._toc_load_generation
        target_dir = self.input_dir

        def worker() -> None:
            text, error = read_toc_text(toc_path)
            GLib.idle_add(self._on_toc_text_loaded, generation, text, error, target_dir)

        threading.Thread(target=worker, daemon=True).start()

    def _on_toc_text_loaded(
        self,
        generation: int,
        text: str,
        error: str | None,
        target_dir: Path,
    ) -> bool:
        if generation != self._toc_load_generation:
            return False
        if target_dir != self.input_dir:
            return False
        if error:
            self._transient_toast(error)
        self._update_toc_from_text(text)
        return False

    def _update_toc_from_text(self, text: str) -> None:
        self._toc_categories = parse_toc_text(text)
        self._rebuild_toc_sidebar()

    def on_toc_text_updated(self, text: str) -> None:
        self._toc_load_generation += 1
        self._update_toc_from_text(text)

    def _nearest_index_for(self, page_num: int) -> int:
        if not self.pages:
            return 0
        pos = bisect.bisect_left(self.pages, page_num)
        if pos == 0:
            return 0
        if pos == len(self.pages):
            return len(self.pages) - 1
        before = self.pages[pos - 1]
        after = self.pages[pos]
        if page_num - before <= after - page_num:
            return pos - 1
        return pos

    def on_activate(self, app: Gio.Application) -> None:  # noqa: ARG002
        self._scan_pages()
        self._ensure_window()
        self._load_toc_from_disk_async()
        self._kickoff_rag_background_load()
        if self.pages:
            self.current_index = 0
            self._load_current()
            self._persist_active_view_state()
        else:
            self._set_window_title("No pages found", "No pages found")
            self._set_text("No .txt pages found in:\n" + str(self.text_dir))
        if self.win:
            self.win.present()

    def _ensure_window(self) -> None:
        if self.win:
            return

        self.win = Adw.ApplicationWindow(application=self)
        self.win.add_css_class("focus-window")
        self.win.set_default_size(900, 700)
        self.win.connect("close-request", self._on_main_window_close_request)
        win_display = self.win.get_display()
        if win_display:
            Gtk.StyleContext.add_provider_for_display(
                win_display,
                _chrome_provider,
                Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
            )

        toolbar = Adw.ToolbarView()
        self.win.set_content(toolbar)

        header = Adw.HeaderBar()
        header.add_css_class("flat")
        header.add_css_class("focus-header")

        left_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        left_box.set_valign(Gtk.Align.CENTER)

        self._toc_sidebar_button = Gtk.ToggleButton()
        self._toc_sidebar_icon = Gtk.Image.new_from_icon_name("sidebar-show-symbolic")
        self._toc_sidebar_icon.add_css_class("focus-toggle-icon")
        self._toc_sidebar_button.set_child(self._toc_sidebar_icon)
        self._toc_sidebar_button.add_css_class("flat")
        self._toc_sidebar_button.set_tooltip_text("Toggle TOC sidebar (Ctrl+Shift+Z)")
        self._toc_sidebar_button.connect("toggled", self._on_sidebar_toggle_button)
        left_box.append(self._toc_sidebar_button)

        self._ai_panel_icon_name = self._choose_icon(*AI_PANEL_ICON_CHOICES)
        self._ai_panel_icon = Gtk.Image.new_from_icon_name(self._ai_panel_icon_name)
        self._ai_panel_icon.add_css_class("focus-toggle-icon")
        self._ai_panel_toggle = Gtk.ToggleButton()
        self._ai_panel_toggle.set_child(self._ai_panel_icon)
        self._ai_panel_toggle.add_css_class("flat")
        self._ai_panel_toggle.set_valign(Gtk.Align.CENTER)
        self._ai_panel_toggle.set_tooltip_text("Show AI panel (Ctrl+Shift+A)")
        self._ai_panel_toggle.connect("toggled", self._on_ai_panel_toggled)
        self._set_ai_panel_visible(False)
        left_box.append(self._ai_panel_toggle)

        header.pack_start(left_box)

        center_wrapper = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        center_wrapper.set_hexpand(True)
        center_wrapper.set_halign(Gtk.Align.CENTER)
        center_wrapper.set_valign(Gtk.Align.CENTER)
        self._center_label = Gtk.Label()
        center_wrapper.append(self._center_label)
        header.set_title_widget(center_wrapper)

        # Hamburger menu on the right
        menu_model = Gio.Menu()
        menu_model.append("Input Directory", "app.choose_input")
        menu_model.append("Create TOC", "app.open_toc")
        menu_model.append("Settings", "app.open_ai_settings")

        view_button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        for view_id in (VIEW_ONE_ID, VIEW_TWO_ID):
            btn = Gtk.ToggleButton(label=VIEW_LABELS.get(view_id, view_id.title()))
            btn.add_css_class("flat")
            btn.add_css_class("no-bold")
            btn.add_css_class("focus-view-toggle")
            btn.set_valign(Gtk.Align.CENTER)
            btn.connect("toggled", self._on_view_button_toggled, view_id)
            self._view_buttons[view_id] = btn
            view_button_box.append(btn)

        menu_button = Gtk.MenuButton(icon_name="open-menu-symbolic")
        menu_button.add_css_class("flat")
        menu_button.set_valign(Gtk.Align.CENTER)
        menu_button.set_popover(Gtk.PopoverMenu.new_from_model(menu_model))

        trailing_header_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        trailing_header_box.set_valign(Gtk.Align.CENTER)
        trailing_header_box.append(view_button_box)
        trailing_header_box.append(menu_button)
        header.pack_end(trailing_header_box)
        self._update_view_buttons()

        # Place headerbar in main window
        toolbar.add_top_bar(header)

        self.textview = Gtk.TextView(editable=False, monospace=False, wrap_mode=Gtk.WrapMode.NONE)
        self.textview.set_hexpand(True)
        self.textview.set_vexpand(True)
        self.textview.set_name("page-text")
        self.textview.set_top_margin(12)
        self.textview.set_bottom_margin(12)
        self.textview.set_left_margin(16)
        self.textview.set_right_margin(16)
        self.textview.set_cursor_visible(False)
        self._apply_text_color(DEFAULT_TEXT_COLOR)
        self._install_textview_link_controllers()

        self.scroller = Gtk.ScrolledWindow()
        self.scroller.add_css_class("focus-scroller")
        self.scroller.set_hexpand(True)
        self.scroller.set_vexpand(True)
        self.scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.scroller.set_propagate_natural_height(False)
        self.scroller.set_min_content_height(0)
        self.scroller.set_size_request(-1, 0)
        self.scroller.set_child(self.textview)

        self._image_picture = Gtk.Picture()
        self._image_picture.set_hexpand(True)
        self._image_picture.set_vexpand(True)
        self._image_picture.set_halign(Gtk.Align.START)
        self._image_picture.set_valign(Gtk.Align.START)
        self._image_picture.set_content_fit(Gtk.ContentFit.CONTAIN)
        self._image_picture.set_can_shrink(True)

        image_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        image_container.set_margin_top(0)
        image_container.set_margin_bottom(0)
        image_container.set_margin_start(0)
        image_container.set_margin_end(0)
        image_container.set_halign(Gtk.Align.FILL)
        image_container.set_valign(Gtk.Align.START)
        image_container.append(self._image_picture)

        self._image_scroller = Gtk.ScrolledWindow()
        self._image_scroller.add_css_class("focus-scroller")
        self._image_scroller.set_hexpand(True)
        self._image_scroller.set_vexpand(True)
        self._image_scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self._image_scroller.set_propagate_natural_height(False)
        self._image_scroller.set_min_content_height(0)
        self._image_scroller.set_size_request(-1, 0)
        self._image_scroller.set_child(image_container)

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        content_box.set_hexpand(True)
        content_box.set_vexpand(True)
        content_box.set_margin_start(12)
        content_box.set_margin_end(12)
        content_box.set_margin_top(5)

        self._continuous_icon_name_on = self._choose_icon(*CONTINUOUS_ICON_ON_CHOICES)
        self._continuous_icon_name_off = self._continuous_icon_name_on
        self._image_icon_name_on = self._choose_icon(*IMAGE_ICON_ON_CHOICES)
        self._image_icon_name_off = self._image_icon_name_on

        self._content_stack = Gtk.Stack()
        self._content_stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        self._content_stack.set_transition_duration(120)
        self._content_stack.set_hhomogeneous(False)
        self._content_stack.set_vhomogeneous(False)
        self._content_stack.add_named(self.scroller, "text")
        self._content_stack.add_named(self._image_scroller, "image")
        self._content_stack.set_visible_child_name("text")

        self._ai_panel_revealer = Gtk.Revealer()
        self._ai_panel_revealer.set_transition_type(Gtk.RevealerTransitionType.SLIDE_DOWN)
        self._ai_panel_revealer.set_reveal_child(False)

        ai_panel_root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        ai_panel_root.set_hexpand(True)
        ai_panel_root.set_vexpand(False)
        ai_panel_root.set_margin_top(12)
        ai_panel_root.set_margin_bottom(0)
        ai_panel_root.set_margin_start(12)
        ai_panel_root.set_margin_end(12)
        ai_panel_root.add_css_class("ai-output-frame")

        ai_header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        ai_header.set_hexpand(True)
        ai_header.set_valign(Gtk.Align.CENTER)

        self._ai_spinner = Gtk.Spinner(spinning=False)
        self._ai_spinner.set_visible(False)
        ai_header.append(self._ai_spinner)

        self._ai_view_stack = Adw.ViewStack()
        try:
            self._ai_view_stack.set_transition_type(Adw.ViewStackTransitionType.CROSSFADE)
        except AttributeError:
            # Older libadwaita versions may not expose transition helpers; fall back to defaults.
            try:
                self._ai_view_stack.set_property("transition-type", Adw.ViewStackTransitionType.CROSSFADE)
            except Exception:
                pass
        self._ai_view_stack.set_hhomogeneous(False)
        self._ai_view_stack.set_vhomogeneous(False)
        self._ai_view_stack.set_vexpand(True)
        self._ai_view_stack.connect("notify::visible-child-name", self._on_ai_view_changed)

        ai_view_toggle_group = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        ai_view_toggle_group.set_hexpand(False)
        ai_view_toggle_group.set_valign(Gtk.Align.CENTER)
        ai_view_toggle_group.add_css_class("linked")
        ai_view_toggle_group.add_css_class("round")

        def add_ai_view_toggle(label: str, view_name: str) -> None:
            button = Gtk.ToggleButton(label=label)
            button.add_css_class("flat")
            button.add_css_class("no-bold")
            button.add_css_class("round")
            button.set_valign(Gtk.Align.CENTER)
            button.connect("toggled", self._on_ai_view_toggle, view_name)
            self._ai_view_toggles[view_name] = button
            ai_view_toggle_group.append(button)

        add_ai_view_toggle("Summarize", AI_VIEW_SUMMARIZE)
        add_ai_view_toggle("Q & A", AI_VIEW_QA)
        add_ai_view_toggle("Show File", AI_VIEW_FILE)

        ai_header.append(ai_view_toggle_group)

        ai_header.append(Gtk.Separator.new(Gtk.Orientation.VERTICAL))

        self._ai_controls_stack = Gtk.Stack()
        self._ai_controls_stack.set_hexpand(True)
        self._ai_controls_stack.set_hhomogeneous(False)
        self._ai_controls_stack.set_vhomogeneous(False)
        self._ai_controls_stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        ai_header.append(self._ai_controls_stack)

        ai_panel_root.append(ai_header)

        summarize_view = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        summarize_view.set_hexpand(True)
        summarize_view.set_vexpand(True)
        summarize_controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        summarize_controls.set_hexpand(True)
        summarize_controls.set_valign(Gtk.Align.CENTER)

        summarize_btn = Gtk.Button(label="Sum Current Page")
        summarize_btn.add_css_class("flat")
        summarize_btn.add_css_class("no-bold")
        summarize_btn.set_valign(Gtk.Align.CENTER)
        summarize_btn.connect("clicked", self._on_summarize_page_clicked)
        summarize_controls.append(summarize_btn)

        self._ai_range_entry = Gtk.Entry()
        self._ai_range_entry.set_placeholder_text("Sum Page Range")
        self._ai_range_entry.set_width_chars(16)
        self._ai_range_entry.set_max_width_chars(18)
        self._ai_range_entry.set_max_length(9)
        self._ai_range_entry.set_hexpand(False)
        self._ai_range_entry.connect("activate", self._on_summarize_range_activate)
        summarize_controls.append(self._ai_range_entry)

        summarize_range_btn = Gtk.Button(label="Submit")
        summarize_range_btn.add_css_class("flat")
        summarize_range_btn.add_css_class("no-bold")
        summarize_range_btn.set_valign(Gtk.Align.CENTER)
        summarize_range_btn.connect("clicked", self._on_summarize_range_button_clicked)
        summarize_controls.append(summarize_range_btn)

        qa_view = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        qa_view.set_hexpand(True)
        qa_view.set_vexpand(True)
        qa_controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        qa_controls.set_hexpand(True)
        qa_controls.set_valign(Gtk.Align.CENTER)

        self._rag_question_entry = Gtk.Entry()
        self._rag_question_entry.set_width_chars(56)
        self._rag_question_entry.set_max_width_chars(72)
        self._rag_question_entry.set_hexpand(True)
        self._rag_question_entry.connect("activate", self._on_rag_question_activate)
        qa_controls.append(self._rag_question_entry)

        ask_button = Gtk.Button(label="Ask")
        ask_button.add_css_class("flat")
        ask_button.add_css_class("no-bold")
        ask_button.set_valign(Gtk.Align.CENTER)
        ask_button.connect("clicked", self._on_rag_question_button_clicked)
        qa_controls.append(ask_button)

        file_view = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        file_view.set_hexpand(True)
        file_view.set_vexpand(True)
        summary_button_group = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        summary_button_group.set_hexpand(True)
        summary_button_group.set_valign(Gtk.Align.CENTER)

        self._choose_summary_button = Gtk.Button(label="Choose File")
        self._choose_summary_button.add_css_class("flat")
        self._choose_summary_button.add_css_class("no-bold")
        self._choose_summary_button.set_valign(Gtk.Align.CENTER)
        self._choose_summary_button.set_tooltip_text("Pick a different text or markdown summary file")
        self._choose_summary_button.connect("clicked", self._on_choose_summary_file_clicked)
        summary_button_group.append(self._choose_summary_button)

        self._summary_search_entry = Gtk.SearchEntry()
        self._summary_search_entry.set_placeholder_text("Search file")
        self._summary_search_entry.set_width_chars(24)
        self._summary_search_entry.set_max_width_chars(48)
        self._summary_search_entry.set_hexpand(True)
        self._summary_search_entry.set_valign(Gtk.Align.CENTER)
        self._summary_search_entry.connect("search-changed", self._on_summary_search_changed)
        self._summary_search_entry.connect("activate", self._on_summary_search_activate)
        summary_button_group.append(self._summary_search_entry)

        self._update_summary_buttons()

        if self._ai_controls_stack:
            self._ai_controls_stack.add_named(summarize_controls, AI_VIEW_SUMMARIZE)
            self._ai_controls_stack.add_named(qa_controls, AI_VIEW_QA)
            self._ai_controls_stack.add_named(summary_button_group, AI_VIEW_FILE)
            self._ai_controls_stack.set_visible_child_name(AI_VIEW_QA)

        self._summary_view = Gtk.TextView(editable=False, monospace=False, wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self._summary_view.add_css_class("ai-output-view")
        self._summary_view.set_hexpand(True)
        self._summary_view.set_vexpand(True)
        self._summary_view.set_top_margin(6)
        self._summary_view.set_bottom_margin(6)
        self._summary_view.set_left_margin(6)
        self._summary_view.set_right_margin(6)
        self._summary_view.set_cursor_visible(False)
        self._summary_view.connect("map", self._on_summary_view_mapped)
        self._summary_buffer = self._summary_view.get_buffer()
        self._install_summary_link_controllers()

        self._summary_scroller = Gtk.ScrolledWindow()
        self._summary_scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self._summary_scroller.set_hexpand(True)
        self._summary_scroller.set_vexpand(True)
        self._summary_scroller.set_propagate_natural_height(True)
        self._summary_scroller.set_min_content_height(AI_OUTPUT_MIN_HEIGHT)
        self._summary_scroller.set_max_content_height(AI_OUTPUT_MAX_HEIGHT)
        self._summary_scroller.set_child(self._summary_view)
        self._connect_summary_scroll_watch()

        file_view.append(self._summary_scroller)

        summarize_scroller = self._build_ai_output_view(AI_VIEW_SUMMARIZE)
        qa_scroller = self._build_ai_output_view(AI_VIEW_QA)

        summarize_view.append(summarize_scroller)
        qa_view.append(qa_scroller)

        self._ai_view_stack.add_titled(summarize_view, AI_VIEW_SUMMARIZE, "Summarize")
        self._ai_view_stack.add_titled(qa_view, AI_VIEW_QA, "Q & A")
        self._ai_view_stack.add_titled(file_view, AI_VIEW_FILE, "Show File")
        self._ai_view_stack.set_visible_child_name(AI_VIEW_QA)
        self._sync_ai_view_toggles(AI_VIEW_QA)

        self._auto_load_summary_file()

        ai_panel_root.append(self._ai_view_stack)
        self._ai_panel_revealer.set_child(ai_panel_root)
        main_root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        main_root.set_hexpand(True)
        main_root.set_vexpand(True)
        main_root.append(self._ai_panel_revealer)

        text_controls = Gtk.CenterBox()
        text_controls.set_hexpand(True)
        text_controls.set_valign(Gtk.Align.CENTER)
        text_controls.set_halign(Gtk.Align.FILL)

        prev_icon_name = self._choose_icon("go-previous-symbolic", "go-previous")
        next_icon_name = self._choose_icon("go-next-symbolic", "go-next")

        def _double_icon(name: str) -> Gtk.Box:
            wrapper = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
            wrapper.set_valign(Gtk.Align.CENTER)
            wrapper.set_halign(Gtk.Align.CENTER)
            wrapper.append(Gtk.Image.new_from_icon_name(name))
            wrapper.append(Gtk.Image.new_from_icon_name(name))
            return wrapper

        paginator = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)

        self._page_back_ten_button = Gtk.Button()
        self._page_back_ten_button.add_css_class("flat")
        self._page_back_ten_button.set_valign(Gtk.Align.CENTER)
        self._page_back_ten_button.set_tooltip_text("Back 10 pages")
        self._page_back_ten_button.set_child(_double_icon(prev_icon_name))
        self._page_back_ten_button.connect("clicked", self._on_page_back_ten_clicked)
        paginator.append(self._page_back_ten_button)

        self._page_back_one_button = Gtk.Button()
        self._page_back_one_button.add_css_class("flat")
        self._page_back_one_button.set_valign(Gtk.Align.CENTER)
        self._page_back_one_button.set_tooltip_text("Previous page (Up)")
        self._page_back_one_button.set_child(Gtk.Image.new_from_icon_name(prev_icon_name))
        self._page_back_one_button.connect("clicked", self._on_page_back_one_clicked)
        paginator.append(self._page_back_one_button)

        self._page_status_label = Gtk.Label(label="--/--")
        self._page_status_label.add_css_class("dim-label")
        self._page_status_label.set_width_chars(10)
        self._page_status_label.set_xalign(0.5)
        paginator.append(self._page_status_label)

        self._page_forward_one_button = Gtk.Button()
        self._page_forward_one_button.add_css_class("flat")
        self._page_forward_one_button.set_valign(Gtk.Align.CENTER)
        self._page_forward_one_button.set_tooltip_text("Next page (Down)")
        self._page_forward_one_button.set_child(Gtk.Image.new_from_icon_name(next_icon_name))
        self._page_forward_one_button.connect("clicked", self._on_page_forward_one_clicked)
        paginator.append(self._page_forward_one_button)

        self._page_forward_ten_button = Gtk.Button()
        self._page_forward_ten_button.add_css_class("flat")
        self._page_forward_ten_button.set_valign(Gtk.Align.CENTER)
        self._page_forward_ten_button.set_tooltip_text("Forward 10 pages")
        self._page_forward_ten_button.set_child(_double_icon(next_icon_name))
        self._page_forward_ten_button.connect("clicked", self._on_page_forward_ten_clicked)
        paginator.append(self._page_forward_ten_button)

        text_controls.set_start_widget(paginator)

        trailing_controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        trailing_controls.set_valign(Gtk.Align.CENTER)
        trailing_controls.set_halign(Gtk.Align.END)

        self._show_image_icon = Gtk.Image.new_from_icon_name(self._image_icon_name_off)
        self._show_image_icon.add_css_class("focus-toggle-icon")
        self._show_image_button = Gtk.ToggleButton()
        self._show_image_button.set_child(self._show_image_icon)
        self._show_image_button.add_css_class("flat")
        self._show_image_button.set_valign(Gtk.Align.CENTER)
        self._show_image_button.set_tooltip_text("Enable image view (Ctrl+I)")
        self._show_image_button.connect("toggled", self._on_show_image_button_toggled)
        trailing_controls.append(self._show_image_button)

        self._continuous_icon = Gtk.Image.new_from_icon_name(self._continuous_icon_name_off)
        self._continuous_icon.add_css_class("focus-toggle-icon")
        self._continuous_toggle_button = Gtk.ToggleButton()
        self._continuous_toggle_button.set_child(self._continuous_icon)
        self._continuous_toggle_button.add_css_class("flat")
        self._continuous_toggle_button.set_valign(Gtk.Align.CENTER)
        self._continuous_toggle_button.set_tooltip_text("Enable continuous view (Ctrl+Shift+C)")
        self._continuous_toggle_button.connect("toggled", self._on_continuous_button_toggled)
        trailing_controls.append(self._continuous_toggle_button)

        self._grep_entry = Gtk.Entry()
        self._grep_entry.set_width_chars(32)
        self._grep_entry.set_max_width_chars(48)
        self._grep_entry.set_hexpand(True)
        self._grep_entry.connect("activate", self._on_grep_entry_activate)
        trailing_controls.append(self._grep_entry)

        grep_search_button = Gtk.Button(label="Search")
        grep_search_button.add_css_class("flat")
        grep_search_button.add_css_class("no-bold")
        grep_search_button.set_valign(Gtk.Align.CENTER)
        grep_search_button.connect("clicked", self._on_grep_search_clicked)
        trailing_controls.append(grep_search_button)

        grep_highlighted_button = Gtk.Button(label="Search Highlighted")
        grep_highlighted_button.add_css_class("flat")
        grep_highlighted_button.add_css_class("no-bold")
        grep_highlighted_button.set_valign(Gtk.Align.CENTER)
        grep_highlighted_button.connect("clicked", self._on_grep_search_highlighted_clicked)
        trailing_controls.append(grep_highlighted_button)
        text_controls.set_end_widget(trailing_controls)

        content_box.append(text_controls)
        self._update_continuous_toggle_button()
        self._update_show_image_toggle_button()
        self._update_page_nav_buttons()

        content_box.append(self._content_stack)

        self._split_view = Adw.NavigationSplitView()
        self._split_view.set_collapsed(False)
        self._split_view.add_css_class("focus-split")
        self._split_view.set_hexpand(True)
        self._split_view.set_vexpand(True)
        self._split_view.set_sidebar_width_fraction(0.05)
        self._split_view.set_min_sidebar_width(180)
        self._split_view.set_max_sidebar_width(320)
        self._split_content_page = Adw.NavigationPage.new(content_box, "Document")
        self._split_view.set_content(self._split_content_page)
        self._split_view.set_collapsed(True)

        sidebar_root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        sidebar_root.set_hexpand(True)
        sidebar_root.set_vexpand(True)
        sidebar_root.add_css_class("focus-sidebar")

        sidebar_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        sidebar_container.set_margin_top(42)
        sidebar_container.set_margin_bottom(8)
        sidebar_container.set_margin_start(0)
        sidebar_container.set_margin_end(0)
        sidebar_container.set_valign(Gtk.Align.FILL)
        sidebar_container.set_vexpand(True)

        self._toc_sidebar_root_store = Gio.ListStore(item_type=FocusSidebarItem)
        self._toc_sidebar_tree_model = Gtk.TreeListModel.new(
            self._toc_sidebar_root_store,
            False,
            False,
            self._create_sidebar_children_model,
        )

        self._toc_list_view = Gtk.ListBox()
        self._toc_list_view.set_selection_mode(Gtk.SelectionMode.NONE)
        self._toc_list_view.add_css_class("focus-sidebar-listview")
        self._toc_list_view.set_activate_on_single_click(True)
        self._toc_list_view.connect("row-activated", self._on_sidebar_row_activated)
        if self._toc_sidebar_tree_model is not None:
            self._toc_list_view.bind_model(self._toc_sidebar_tree_model, self._create_sidebar_row_for_item)

        self._toc_placeholder = Gtk.Label(label="No TOC loaded", xalign=0)
        self._toc_placeholder.add_css_class("dim-label")
        self._toc_placeholder.set_hexpand(True)
        self._toc_placeholder.set_vexpand(True)
        self._toc_placeholder.set_margin_top(24)
        self._toc_placeholder.set_margin_start(4)
        self._toc_placeholder.set_margin_end(4)
        self._toc_placeholder.set_halign(Gtk.Align.START)
        self._toc_placeholder.set_valign(Gtk.Align.START)

        self._toc_sidebar_overlay = Gtk.Overlay()
        self._toc_sidebar_overlay.set_child(self._toc_list_view)
        self._toc_sidebar_overlay.add_overlay(self._toc_placeholder)
        self._toc_sidebar_overlay.set_hexpand(True)
        self._toc_sidebar_overlay.set_vexpand(True)
        self._toc_sidebar_overlay.set_valign(Gtk.Align.START)

        self._toc_sidebar_scroller = Gtk.ScrolledWindow()
        self._toc_sidebar_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._toc_sidebar_scroller.set_hexpand(True)
        self._toc_sidebar_scroller.set_vexpand(True)
        self._toc_sidebar_scroller.set_propagate_natural_height(False)
        self._toc_sidebar_scroller.set_child(self._toc_sidebar_overlay)

        sidebar_container.append(self._toc_sidebar_scroller)

        self._toc_sidebar_revealer = Gtk.Revealer()
        self._toc_sidebar_revealer.set_transition_type(Gtk.RevealerTransitionType.SLIDE_RIGHT)
        self._toc_sidebar_revealer.set_reveal_child(self._toc_sidebar_visible)
        sidebar_root.append(sidebar_container)
        self._toc_sidebar_revealer.set_child(sidebar_root)
        self._split_sidebar_page = Adw.NavigationPage.new(self._toc_sidebar_revealer, "Contents")

        main_root.append(self._split_view)
        toolbar.set_content(main_root)
        self._rebuild_toc_sidebar()

        self._install_navigation_controllers()
        self._install_actions()

    def _get_view_state(self, view_id: str) -> FocusViewState:
        state = self._views.get(view_id)
        if state is None:
            state = FocusViewState(name=VIEW_LABELS.get(view_id, view_id))
            self._views[view_id] = state
        return state

    def _current_view_state(self) -> FocusViewState:
        return self._get_view_state(self._active_view_id)

    def _reset_view_states(self) -> None:
        self._cancel_all_ai_streams()
        self._views = {
            VIEW_ONE_ID: FocusViewState(name=VIEW_LABELS[VIEW_ONE_ID]),
            VIEW_TWO_ID: FocusViewState(name=VIEW_LABELS[VIEW_TWO_ID]),
        }
        self._active_view_id = VIEW_ONE_ID
        self._current_view_state().sidebar_visible = self._toc_sidebar_visible
        self._current_view_state().ai_panel_visible = bool(
            (self._ai_panel_toggle and self._ai_panel_toggle.get_active())
            or (self._ai_panel_revealer and self._ai_panel_revealer.get_child_revealed())
        )
        self._ai_active_view = AI_VIEW_QA
        self._ai_request_generation = 0
        self._ai_in_flight = False
        self._ai_cancel_event = None
        self._ai_stream_thread = None
        for ai_state in self._ai_outputs.values():
            ai_state.raw = ""
            self._apply_ai_output_links("", ai_state)
        self._sync_continuous_action()
        self._sync_show_image_action()
        self._update_view_buttons()

    def _cancel_all_ai_streams(self) -> None:
        for state in self._views.values():
            if state.ai_cancel_event:
                state.ai_cancel_event.set()
            if state.ai_stream_thread and state.ai_stream_thread.is_alive():
                try:
                    state.ai_stream_thread.join(timeout=0.2)
                except Exception:
                    pass
            state.ai_in_flight = False
            state.ai_cancel_event = None
            state.ai_stream_thread = None
        if self._active_view_id in self._views:
            self._ai_in_flight = False
            self._ai_cancel_event = None
            self._ai_stream_thread = None

    def _persist_active_view_state(self) -> None:
        state = self._current_view_state()
        state.current_index = self.current_index
        state.show_image = self._show_image
        state.sidebar_visible = self._toc_sidebar_visible
        state.ai_panel_visible = bool(
            (self._ai_panel_toggle and self._ai_panel_toggle.get_active())
            or (self._ai_panel_revealer and self._ai_panel_revealer.get_child_revealed())
        )
        state.continuous_view = self._continuous_view
        state.continuous_text = self._continuous_text
        state.continuous_pages_order = list(self._continuous_pages_order)
        state.continuous_loaded_count = self._continuous_loaded_count
        state.continuous_loading = self._continuous_loading
        state.grep_phrase_raw = self._grep_phrase_raw
        state.grep_regex = self._grep_regex
        state.grep_hits = {k: list(v) for k, v in self._grep_hits.items()}
        state.matching_pages = list(self._matching_pages)
        state.matching_lookup = dict(self._matching_lookup)
        state.grep_combined_text = self._grep_combined_text
        state.grep_combined_highlights = list(self._grep_combined_highlights)
        state.showing_grep_results = self._showing_grep_results
        state.ai_active_view = self._ai_active_view
        state.ai_output_raw = {name: view.raw or "" for name, view in self._ai_outputs.items()}
        state.ai_status_text = ""
        state.ai_spinning = bool(self._ai_spinner and self._ai_spinner.get_spinning())
        state.ai_request_generation = self._ai_request_generation
        state.ai_in_flight = self._ai_in_flight
        state.ai_cancel_event = self._ai_cancel_event
        state.ai_stream_thread = self._ai_stream_thread
        state.sidebar_expanded = self._get_sidebar_expanded_keys()
        if self._ai_range_entry:
            state.ai_range_text = self._ai_range_entry.get_text()
        if self._rag_question_entry:
            state.rag_question_text = self._rag_question_entry.get_text()

    def _apply_ai_outputs_for_state(self, state: FocusViewState) -> None:
        for name, view in self._ai_outputs.items():
            raw_text = state.ai_output_raw.get(name, "")
            view.raw = raw_text
            self._apply_ai_output_links(raw_text, view)

    def _restore_view_state(self, state: FocusViewState) -> None:
        self._disconnect_continuous_scroll_watch()
        self.current_index = state.current_index
        self._continuous_view = state.continuous_view
        self._show_image = state.show_image and not self._continuous_view
        self._continuous_text = state.continuous_text
        self._continuous_pages_order = list(state.continuous_pages_order)
        self._continuous_loaded_count = state.continuous_loaded_count
        self._continuous_loading = False
        self._grep_phrase_raw = state.grep_phrase_raw
        self._grep_regex = state.grep_regex
        self._grep_hits = {k: list(v) for k, v in state.grep_hits.items()}
        self._matching_pages = list(state.matching_pages)
        self._matching_lookup = dict(state.matching_lookup)
        self._grep_combined_text = state.grep_combined_text
        self._grep_combined_highlights = list(state.grep_combined_highlights)
        self._showing_grep_results = state.showing_grep_results
        self._ai_active_view = state.ai_active_view
        self._ai_request_generation = state.ai_request_generation
        self._ai_in_flight = state.ai_in_flight
        self._ai_cancel_event = state.ai_cancel_event
        self._ai_stream_thread = state.ai_stream_thread
        self._set_sidebar_visible(state.sidebar_visible)
        self._set_ai_panel_visible(state.ai_panel_visible)
        if self._grep_entry:
            self._grep_entry.set_text(self._grep_phrase_raw or "")
        if self._ai_range_entry:
            self._ai_range_entry.set_text(state.ai_range_text)
        if self._rag_question_entry:
            self._rag_question_entry.set_text(state.rag_question_text)
        self._set_ai_view(state.ai_active_view)
        self._apply_ai_outputs_for_state(state)
        if self._ai_spinner:
            self._ai_spinner.set_spinning(state.ai_spinning)
            self._ai_spinner.set_visible(state.ai_spinning)
        self._update_continuous_toggle_button()
        self._update_show_image_toggle_button()
        self._sync_continuous_action()
        self._sync_show_image_action()
        self._update_view_buttons()
        self._apply_sidebar_expansion_state(state)
        if self._showing_grep_results and self._grep_combined_text:
            self._set_show_image(False, silent=True)
            self._show_grep_results()
            return
        if self._continuous_view:
            self._show_image_update_visible()
            if self._continuous_text:
                self._set_text(self._continuous_text, None)
                self._connect_continuous_scroll_watch()
                self._update_header()
            else:
                self._set_continuous_view(True)
            return
        self._set_show_image(self._show_image, silent=True)
        self._load_current()

    def _switch_view(self, view_id: str) -> None:
        if view_id == self._active_view_id or view_id not in self._views:
            self._update_view_buttons()
            return
        self._persist_active_view_state()
        self._active_view_id = view_id
        self._update_view_buttons()
        self._restore_view_state(self._current_view_state())

    def _update_view_buttons(self) -> None:
        if not self._view_buttons:
            return
        self._view_button_guard = True
        try:
            for view_id, button in self._view_buttons.items():
                desired = view_id == self._active_view_id
                if button.get_active() != desired:
                    button.set_active(desired)
        finally:
            self._view_button_guard = False

    def _on_view_button_toggled(self, button: Gtk.ToggleButton, view_id: str) -> None:
        if self._view_button_guard:
            return
        if not button.get_active():
            self._update_view_buttons()
            return
        self._switch_view(view_id)

    def _set_window_title(
        self,
        subtitle: str | None = None,
        window_suffix: str | None = None,
    ) -> None:
        if self.win:
            title = "Focus"
            if window_suffix:
                title = f"{title} - {window_suffix}"
            self.win.set_title(title)
        if self._center_label:
            label_markup = "<b>Focus</b>"
            if subtitle:
                safe_subtitle = GLib.markup_escape_text(subtitle)
                label_markup = f"{label_markup} - {safe_subtitle}"
            self._center_label.set_markup(label_markup)

    def _set_text(self, text: str, highlights: list[tuple[int, int]] | None = None) -> None:
        if not self.textview:
            return
        buf = self.textview.get_buffer()
        buf.set_text(text)
        self._apply_page_links(buf, text)
        if highlights:
            tag = self._ensure_highlight_tag()
            if tag is not None:
                char_count = buf.get_char_count()
                for start, end in highlights:
                    if end <= start:
                        continue
                    start = max(0, min(start, char_count))
                    end = max(0, min(end, char_count))
                    if end <= start:
                        continue
                    start_iter = buf.get_iter_at_offset(start)
                    end_iter = buf.get_iter_at_offset(end)
                    buf.apply_tag(tag, start_iter, end_iter)
        self._apply_keyword_highlights(buf, text)
        if self.scroller:
            vadj = self.scroller.get_vadjustment()
            if vadj:
                GLib.idle_add(vadj.set_value, vadj.get_lower())
            hadj = self.scroller.get_hadjustment()
            if hadj:
                GLib.idle_add(hadj.set_value, hadj.get_lower())

    def _rebuild_toc_sidebar(self) -> None:
        if self._toc_sidebar_root_store is None:
            return
        self._toc_sidebar_root_store.remove_all()
        if not self._toc_categories:
            self._update_sidebar_placeholder(False)
            self._sync_sidebar_active_page()
            return
        for category in self._toc_categories:
            item = FocusSidebarItem.from_category(category)
            self._toc_sidebar_root_store.append(item)
        self._update_sidebar_placeholder(True)
        self._apply_sidebar_expansion_state(self._current_view_state())
        self._sync_sidebar_active_page()

    def _create_sidebar_children_model(self, item: GObject.Object) -> Gio.ListModel | None:
        if isinstance(item, FocusSidebarItem):
            return item.get_children_model()
        return None

    def _create_sidebar_row_for_item(self, obj: GObject.Object, _user_data: object | None = None) -> Gtk.Widget:
        row_widget = Gtk.ListBoxRow()
        row_widget.set_activatable(True)
        row_widget.add_css_class("focus-sidebar-listbox-row")
        action_row = Adw.ActionRow()
        action_row.add_css_class("flat")
        action_row.add_css_class("focus-sidebar-row")
        action_row.set_activatable(False)
        action_row.set_hexpand(True)
        arrow_icon = Gtk.Image.new_from_icon_name("pan-end-symbolic")
        arrow_button = Gtk.ToggleButton()
        arrow_button.add_css_class("focus-sidebar-expand-button")
        arrow_button.set_child(arrow_icon)
        arrow_button.set_focus_on_click(False)
        arrow_button.set_valign(Gtk.Align.CENTER)
        arrow_button.set_visible(False)
        arrow_button._focus_list_item = row_widget  # type: ignore[attr-defined]
        arrow_button.connect("toggled", self._on_sidebar_expand_button_toggled)
        action_row.add_prefix(arrow_button)
        row_widget.set_child(action_row)
        row_widget._focus_row = action_row  # type: ignore[attr-defined]
        row_widget._focus_arrow_button = arrow_button  # type: ignore[attr-defined]
        row_widget._focus_arrow_icon = arrow_icon  # type: ignore[attr-defined]
        row_widget._focus_arrow_guard = False  # type: ignore[attr-defined]
        row_widget._focus_tree_row = None  # type: ignore[attr-defined]
        row_widget._focus_tree_handler = None  # type: ignore[attr-defined]
        row_widget.connect("destroy", self._on_sidebar_row_destroy)
        if isinstance(obj, Gtk.TreeListRow):
            self._bind_sidebar_row(row_widget, obj)
        return row_widget

    def _bind_sidebar_row(self, list_row: Gtk.ListBoxRow, tree_row: Gtk.TreeListRow) -> None:
        action_row = getattr(list_row, "_focus_row", None)
        arrow_button = getattr(list_row, "_focus_arrow_button", None)
        arrow_icon = getattr(list_row, "_focus_arrow_icon", None)
        if action_row is None or arrow_button is None or arrow_icon is None:
            return
        previous_row = getattr(list_row, "_focus_tree_row", None)
        previous_handler = getattr(list_row, "_focus_tree_handler", None)
        if isinstance(previous_row, Gtk.TreeListRow) and isinstance(previous_handler, int):
            try:
                previous_row.disconnect(previous_handler)
            except (TypeError, RuntimeError):
                pass
        list_row._focus_tree_row = tree_row  # type: ignore[attr-defined]
        handler_id = tree_row.connect("notify::expanded", self._on_sidebar_tree_row_expanded, list_row)
        list_row._focus_tree_handler = handler_id  # type: ignore[attr-defined]
        depth = max(tree_row.get_depth(), 0)
        action_row.set_margin_start(depth * SIDEBAR_TREE_INDENT)
        item = tree_row.get_item()
        if not isinstance(item, FocusSidebarItem):
            action_row.set_title("")
            action_row.set_subtitle("")
            arrow_button.set_visible(False)
            return
        action_row.remove_css_class("focus-sidebar-category")
        action_row.remove_css_class("focus-sidebar-bookmark")
        action_row.remove_css_class("focus-sidebar-category-expanded")
        if item.kind == "category":
            action_row.add_css_class("focus-sidebar-category")
        else:
            action_row.add_css_class("focus-sidebar-bookmark")
        action_row.set_title(item.title)
        if item.kind == "bookmark" and item.page is not None:
            action_row.set_subtitle(f"Page {item.page:04d}")
        else:
            action_row.set_subtitle("")
        self._update_sidebar_row_expand_widgets(list_row, tree_row)
        self._update_sidebar_row_active_state(list_row)

    def _on_sidebar_row_destroy(self, list_row: Gtk.ListBoxRow) -> None:
        tree_row = getattr(list_row, "_focus_tree_row", None)
        handler_id = getattr(list_row, "_focus_tree_handler", None)
        if isinstance(tree_row, Gtk.TreeListRow) and isinstance(handler_id, int):
            try:
                tree_row.disconnect(handler_id)
            except (TypeError, RuntimeError):
                pass
        list_row._focus_tree_row = None  # type: ignore[attr-defined]
        list_row._focus_tree_handler = None  # type: ignore[attr-defined]

    def _on_sidebar_row_activated(self, _list_box: Gtk.ListBox, row: Gtk.ListBoxRow) -> None:
        tree_row = getattr(row, "_focus_tree_row", None)
        if not isinstance(tree_row, Gtk.TreeListRow):
            return
        item = tree_row.get_item()
        if not isinstance(item, FocusSidebarItem):
            return
        if item.kind == "category":
            tree_row.set_expanded(not tree_row.get_expanded())
            return
        self._show_page_from_link(f"{item.page:04d}")

    def _update_sidebar_row_active_state(self, list_row: Gtk.ListBoxRow) -> None:
        action_row = getattr(list_row, "_focus_row", None)
        tree_row = getattr(list_row, "_focus_tree_row", None)
        if not isinstance(action_row, Adw.ActionRow) or not isinstance(tree_row, Gtk.TreeListRow):
            return
        action_row.remove_css_class("focus-sidebar-bookmark-active")
        if not self.pages or not (0 <= self.current_index < len(self.pages)):
            return
        current_page = self.pages[self.current_index]
        item = tree_row.get_item()
        if (
            isinstance(item, FocusSidebarItem)
            and item.kind == "bookmark"
            and item.page == current_page
        ):
            action_row.add_css_class("focus-sidebar-bookmark-active")

    def _sync_sidebar_active_page(self) -> None:
        if not self._toc_list_view:
            return
        child = self._toc_list_view.get_first_child()
        while child:
            if isinstance(child, Gtk.ListBoxRow):
                self._update_sidebar_row_active_state(child)
            child = child.get_next_sibling()

    def _update_sidebar_placeholder(self, has_items: bool) -> None:
        self._toc_sidebar_has_items = has_items
        if self._toc_list_view is not None:
            self._toc_list_view.set_sensitive(has_items)
        if self._toc_placeholder is not None:
            self._toc_placeholder.set_visible(not has_items)

    def _update_sidebar_row_expand_widgets(self, list_row: Gtk.ListBoxRow, tree_row: Gtk.TreeListRow) -> None:
        arrow_button = getattr(list_row, "_focus_arrow_button", None)
        arrow_icon = getattr(list_row, "_focus_arrow_icon", None)
        row = getattr(list_row, "_focus_row", None)
        if arrow_button is None or arrow_icon is None:
            return
        if tree_row.is_expandable():
            arrow_button.set_visible(True)
            list_row._focus_arrow_guard = True  # type: ignore[attr-defined]
            try:
                arrow_button.set_active(tree_row.get_expanded())
            finally:
                list_row._focus_arrow_guard = False  # type: ignore[attr-defined]
            expanded = tree_row.get_expanded()
            icon_name = "pan-down-symbolic" if expanded else "pan-end-symbolic"
            arrow_icon.set_from_icon_name(icon_name)
            if isinstance(row, Adw.ActionRow):
                if expanded:
                    row.add_css_class("focus-sidebar-category-expanded")
                else:
                    row.remove_css_class("focus-sidebar-category-expanded")
        else:
            arrow_button.set_visible(False)
            if isinstance(row, Adw.ActionRow):
                row.remove_css_class("focus-sidebar-category-expanded")

    def _on_sidebar_expand_button_toggled(self, button: Gtk.ToggleButton) -> None:
        list_row = getattr(button, "_focus_list_item", None)
        if list_row is None or getattr(list_row, "_focus_arrow_guard", False):
            return
        tree_row = getattr(list_row, "_focus_tree_row", None)
        arrow_icon = getattr(list_row, "_focus_arrow_icon", None)
        if not isinstance(tree_row, Gtk.TreeListRow):
            return
        tree_row.set_expanded(button.get_active())
        if arrow_icon:
            icon_name = "pan-down-symbolic" if button.get_active() else "pan-end-symbolic"
            arrow_icon.set_from_icon_name(icon_name)

    def _on_sidebar_tree_row_expanded(
        self,
        tree_row: Gtk.TreeListRow,
        _pspec: GObject.ParamSpec,
        list_row: Gtk.ListBoxRow,
    ) -> None:
        self._update_sidebar_row_expand_widgets(list_row, tree_row)
        self._current_view_state().sidebar_expanded = self._get_sidebar_expanded_keys()

    def _sidebar_item_key(self, item: FocusSidebarItem) -> str:
        page = "" if item.page is None else str(item.page)
        return f"{item.title}::{page}"

    def _get_sidebar_expanded_keys(self) -> list[str]:
        if not self._toc_list_view:
            return []
        keys: list[str] = []
        seen: set[str] = set()
        child = self._toc_list_view.get_first_child()
        while child:
            if isinstance(child, Gtk.ListBoxRow):
                tree_row = getattr(child, "_focus_tree_row", None)
                if isinstance(tree_row, Gtk.TreeListRow):
                    item = tree_row.get_item()
                    if (
                        isinstance(item, FocusSidebarItem)
                        and item.kind == "category"
                        and tree_row.get_expanded()
                    ):
                        key = self._sidebar_item_key(item)
                        if key not in seen:
                            seen.add(key)
                            keys.append(key)
            child = child.get_next_sibling()
        return keys

    def _apply_sidebar_expansion_state(self, state: FocusViewState) -> None:
        if not self._toc_list_view:
            return
        desired = set(state.sidebar_expanded)
        rows: list[tuple[Gtk.TreeListRow, FocusSidebarItem]] = []
        child = self._toc_list_view.get_first_child()
        while child:
            if isinstance(child, Gtk.ListBoxRow):
                tree_row = getattr(child, "_focus_tree_row", None)
                if isinstance(tree_row, Gtk.TreeListRow):
                    item = tree_row.get_item()
                    if isinstance(item, FocusSidebarItem) and item.kind == "category":
                        rows.append((tree_row, item))
            child = child.get_next_sibling()
        for tree_row, item in rows:
            should_expand = self._sidebar_item_key(item) in desired
            if tree_row.get_expanded() != should_expand:
                tree_row.set_expanded(should_expand)

    def _build_continuous_document(self) -> str:
        ordered = self._continuous_page_order()
        return self._render_continuous_chunk(ordered)

    def _continuous_page_order(self) -> list[int]:
        if not self.pages:
            return []
        return self.pages[self.current_index :] + self.pages[: self.current_index]

    def _render_continuous_chunk(self, ordered: list[int]) -> str:
        parts: list[str] = []
        for idx, page in enumerate(ordered):
            content, _, _ = self._read_page_text(page)
            rendered, _ = self._render_page_display(page, content, None)
            parts.append(rendered)
            if idx != len(ordered) - 1:
                parts.append("\n\n")
        return "".join(parts)

    def _connect_continuous_scroll_watch(self) -> None:
        if not self.scroller:
            return
        vadj = self.scroller.get_vadjustment()
        if not vadj:
            return
        self._disconnect_continuous_scroll_watch()
        self._continuous_vadj_handler = vadj.connect("value-changed", self._on_continuous_scroll)

    def _disconnect_continuous_scroll_watch(self) -> None:
        if self._continuous_vadj_handler is None or not self.scroller:
            self._continuous_vadj_handler = None
            return
        vadj = self.scroller.get_vadjustment()
        if vadj:
            try:
                vadj.disconnect(self._continuous_vadj_handler)
            except (TypeError, RuntimeError):
                pass
        self._continuous_vadj_handler = None

    def _on_continuous_scroll(self, adjustment: Gtk.Adjustment) -> None:
        if not self._continuous_view:
            return
        self._maybe_load_more_continuous_pages(adjustment)

    def _maybe_load_more_continuous_pages(self, adjustment: Gtk.Adjustment | None = None) -> None:
        if (
            not self._continuous_view
            or self._continuous_loaded_count >= len(self._continuous_pages_order)
            or self._continuous_loading
        ):
            return
        vadj = adjustment
        if not vadj and self.scroller:
            vadj = self.scroller.get_vadjustment()
        if not vadj:
            return
        upper = vadj.get_upper()
        page_size = vadj.get_page_size()
        value = vadj.get_value()
        remaining = upper - (value + page_size)
        if remaining > CONTINUOUS_SCROLL_THRESHOLD_PX:
            return
        self._load_next_continuous_batch(initial=False)

    def _load_next_continuous_batch(self, *, initial: bool) -> None:
        if (
            not self.textview
            or not self._continuous_pages_order
            or self._continuous_loaded_count >= len(self._continuous_pages_order)
        ):
            return
        if self._continuous_loading:
            return
        self._continuous_loading = True
        try:
            start = self._continuous_loaded_count
            end = min(len(self._continuous_pages_order), start + CONTINUOUS_PAGE_BATCH)
            chunk_pages = self._continuous_pages_order[start:end]
            chunk = self._render_continuous_chunk(chunk_pages)
            self._continuous_loaded_count = end
            if not chunk:
                return
            if initial:
                self._continuous_text = chunk
                self._set_text(chunk, None)
                self._update_header()
                return
            buf = self.textview.get_buffer()
            if not buf:
                return
            start_offset = buf.get_char_count()
            prefix = "\n\n" if start_offset > 0 else ""
            text_to_insert = prefix + chunk
            buf.insert(buf.get_end_iter(), text_to_insert)
            self._continuous_text = (self._continuous_text or "") + text_to_insert
            chunk_offset = start_offset + len(prefix)
            self._append_page_links(buf, chunk, chunk_offset)
        finally:
            self._continuous_loading = False

    def _sync_continuous_action(self) -> None:
        if not self._continuous_action:
            self._update_continuous_toggle_button()
            return
        state = self._continuous_action.get_state()
        current = state.get_boolean() if state is not None else None
        if current != self._continuous_view:
            self._continuous_action.set_state(GLib.Variant.new_boolean(self._continuous_view))
        self._update_continuous_toggle_button()

    def _update_continuous_toggle_button(self) -> None:
        if not self._continuous_toggle_button or not self._continuous_icon:
            return
        self._continuous_button_guard = True
        try:
            self._continuous_toggle_button.set_active(self._continuous_view)
        finally:
            self._continuous_button_guard = False
        icon_name = (
            self._continuous_icon_name_on if self._continuous_view else self._continuous_icon_name_off
        )
        self._continuous_icon.set_from_icon_name(icon_name)
        tooltip = (
            "Disable continuous view (Ctrl+Shift+C)"
            if self._continuous_view
            else "Enable continuous view (Ctrl+Shift+C)"
        )
        self._continuous_toggle_button.set_tooltip_text(tooltip)

    def _deactivate_continuous_view(self, *, reload: bool) -> None:
        if not self._continuous_view:
            return
        self._continuous_view = False
        self._continuous_text = None
        self._continuous_pages_order = []
        self._continuous_loaded_count = 0
        self._continuous_loading = False
        self._disconnect_continuous_scroll_watch()
        self._sync_continuous_action()
        def cleanup() -> bool:
            if reload:
                self._load_current()
            else:
                self._update_header()
            self.textview.set_hexpand(True)
            self.textview.set_vexpand(True)
            return False
        GLib.idle_add(cleanup)

    def _set_continuous_view(self, enabled: bool) -> bool:
        if enabled:
            if self._showing_grep_results:
                self._transient_toast("Continuous view is unavailable while showing grep results.")
                self._sync_continuous_action()
                return False
            if not self.pages:
                self._transient_toast("No pages available for continuous view.")
                self._sync_continuous_action()
                return False
            self._set_show_image(False, silent=True)
            self._continuous_pages_order = self._continuous_page_order()
            self._continuous_loaded_count = 0
            self._continuous_text = None
            if not self._continuous_pages_order:
                self._sync_continuous_action()
                return False
            self._continuous_view = True
            self._sync_continuous_action()
            self._load_next_continuous_batch(initial=True)
            self._connect_continuous_scroll_watch()
            return True
        self._deactivate_continuous_view(reload=True)
        return True

    def _set_sidebar_visible(self, visible: bool) -> None:
        self._toc_sidebar_visible = visible
        self._current_view_state().sidebar_visible = visible
        if self._split_view:
            self._split_view.set_collapsed(not visible)
            if visible and self._split_sidebar_page:
                self._split_view.set_sidebar(self._split_sidebar_page)
            else:
                self._split_view.set_sidebar(None)
        if self._toc_sidebar_revealer:
            self._toc_sidebar_revealer.set_reveal_child(visible)
        self._sync_sidebar_controls()

    def _sync_sidebar_controls(self) -> None:
        if self._toc_sidebar_button and self._toc_sidebar_button.get_active() != self._toc_sidebar_visible:
            self._sidebar_button_guard = True
            self._toc_sidebar_button.set_active(self._toc_sidebar_visible)
            self._sidebar_button_guard = False
        if self._toc_sidebar_icon:
            self._toc_sidebar_icon.set_from_icon_name("sidebar-show-symbolic")
        if self._toc_sidebar_button:
            tooltip = (
                "Hide TOC sidebar (Ctrl+Shift+Z)"
                if self._toc_sidebar_visible
                else "Show TOC sidebar (Ctrl+Shift+Z)"
            )
            self._toc_sidebar_button.set_tooltip_text(tooltip)
        if self._toc_sidebar_action:
            state = self._toc_sidebar_action.get_state()
            current = state.get_boolean() if state is not None else None
            if current != self._toc_sidebar_visible:
                self._toc_sidebar_action.set_state(GLib.Variant.new_boolean(self._toc_sidebar_visible))

    def _on_sidebar_toggle_button(self, button: Gtk.ToggleButton) -> None:
        if self._sidebar_button_guard:
            return
        self._set_sidebar_visible(button.get_active())

    def _on_toggle_toc_sidebar(
        self,
        action: Gio.SimpleAction,
        value: GLib.Variant,
    ) -> None:
        visible = value.get_boolean()
        action.set_state(value)
        self._set_sidebar_visible(visible)

    def _on_toggle_continuous_view(
        self,
        action: Gio.SimpleAction,
        value: GLib.Variant,
    ) -> None:
        desired = value.get_boolean()
        success = self._set_continuous_view(desired)
        if not success:
            action.set_state(GLib.Variant.new_boolean(self._continuous_view))
        else:
            action.set_state(GLib.Variant.new_boolean(self._continuous_view))

    def _on_continuous_button_toggled(self, button: Gtk.ToggleButton) -> None:
        if self._continuous_button_guard:
            return
        desired = button.get_active()
        self._set_continuous_view(desired)
        self._update_continuous_toggle_button()

    def _on_toggle_show_image(
        self,
        action: Gio.SimpleAction,
        value: GLib.Variant,
    ) -> None:
        desired = value.get_boolean()
        self._set_show_image(desired)
        action.set_state(GLib.Variant.new_boolean(self._show_image))

    def _on_show_image_button_toggled(self, button: Gtk.ToggleButton) -> None:
        if self._show_image_button_guard:
            return
        desired = button.get_active()
        self._set_show_image(desired)
        self._update_show_image_toggle_button()

    def _apply_text_color(self, color_value: str) -> None:
        self._current_text_color = color_value
        css = (
            "#page-text { "
            f"color: {color_value}; font-size: {self._font_size_pt}pt; "
            "}"
            "textview.ai-output-view { "
            f"color: {color_value}; font-size: {self._ai_font_size_pt}pt; line-height: {AI_OUTPUT_LINE_HEIGHT}; "
            "}"
        ).encode()
        try:
            self._color_provider.load_from_data(css)
        except GLib.Error:
            return
        self._ensure_color_provider()

    def _ensure_color_provider(self) -> None:
        if self._css_provider_registered:
            return
        display = Gdk.Display.get_default()
        if not display:
            return
        Gtk.StyleContext.add_provider_for_display(
            display,
            self._color_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )
        self._css_provider_registered = True

    def _clear_page_links(self, table: Gtk.TextTagTable | None) -> None:
        if table is None:
            return
        for existing_tag in self._link_tags:
            try:
                table.remove(existing_tag)
            except TypeError:
                pass
        self._link_tags.clear()
        self._link_tag_lookup.clear()

    def _ensure_highlight_tag(self) -> Gtk.TextTag | None:
        if not self.textview:
            return None
        buf = self.textview.get_buffer()
        table = buf.get_tag_table()
        tag = table.lookup("match-highlight") if table is not None else None
        if tag is None:
            tag = buf.create_tag("match-highlight", foreground=DEFAULT_MATCH_COLOR)
        return tag

    def _ensure_keyword_highlight_tag(self) -> Gtk.TextTag | None:
        if not self.textview:
            return None
        buf = self.textview.get_buffer()
        table = buf.get_tag_table()
        tag = table.lookup("keyword-highlight") if table is not None else None
        if tag is None:
            tag = buf.create_tag("keyword-highlight", foreground=DEFAULT_HIGHLIGHT_COLOR)
        return tag

    def _apply_keyword_highlights(self, buf: Gtk.TextBuffer, text: str) -> None:
        phrases = self._ai_settings.highlight_phrases if self._ai_settings else []
        if not phrases:
            return
        tag = self._ensure_keyword_highlight_tag()
        if tag is None:
            return
        char_count = buf.get_char_count()
        for phrase in phrases:
            if not phrase:
                continue
            start = 0
            phrase_len = len(phrase)
            while start < len(text):
                idx = text.find(phrase, start)
                if idx == -1:
                    break
                end = idx + phrase_len
                if end <= idx:
                    start = idx + 1
                    continue
                if idx < char_count:
                    end = min(end, char_count)
                    start_iter = buf.get_iter_at_offset(idx)
                    end_iter = buf.get_iter_at_offset(end)
                    buf.apply_tag(tag, start_iter, end_iter)
                start = end

    def _append_page_links(self, buf: Gtk.TextBuffer, text: str, start_offset: int) -> None:
        table = buf.get_tag_table()
        if table is None:
            return
        for match in PAGE_HEADER_LINE_RE.finditer(text):
            page_str = match.group("num")
            if not page_str:
                continue
            start = start_offset + match.start("num")
            end = start_offset + match.end("num")
            start_iter = buf.get_iter_at_offset(start)
            end_iter = buf.get_iter_at_offset(end)
            page_tag = buf.create_tag(
                None,
                foreground="#62a0ea",
                underline=Pango.Underline.SINGLE,
            )
            self._link_tag_lookup[page_tag] = ("page", page_str)
            buf.apply_tag(page_tag, start_iter, end_iter)
            self._link_tags.append(page_tag)

    def _apply_page_links(self, buf: Gtk.TextBuffer, text: str) -> None:
        table = buf.get_tag_table()
        if table is None:
            return
        self._clear_page_links(table)
        self._append_page_links(buf, text, 0)

    def _get_ai_output_state(self, view_name: str) -> AiOutputView:
        state = self._ai_outputs.get(view_name)
        if state is None:
            state = AiOutputView()
            self._ai_outputs[view_name] = state
        return state

    def _build_ai_output_view(self, view_name: str) -> Gtk.ScrolledWindow:
        state = self._get_ai_output_state(view_name)
        text_view = Gtk.TextView(editable=False, monospace=False, wrap_mode=Gtk.WrapMode.WORD_CHAR)
        text_view.add_css_class("ai-output-view")
        text_view.set_hexpand(True)
        text_view.set_vexpand(True)
        text_view.set_top_margin(6)
        text_view.set_bottom_margin(6)
        text_view.set_left_margin(6)
        text_view.set_right_margin(6)
        text_view.set_cursor_visible(False)
        text_view.connect("map", self._on_ai_output_view_mapped, view_name)
        state.view = text_view
        state.buffer = text_view.get_buffer()
        self._install_ai_output_link_controllers(state)

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroller.set_hexpand(True)
        scroller.set_vexpand(True)
        scroller.set_propagate_natural_height(True)
        scroller.set_min_content_height(AI_OUTPUT_MIN_HEIGHT)
        scroller.set_max_content_height(AI_OUTPUT_MAX_HEIGHT)
        scroller.set_child(text_view)
        state.scroller = scroller
        return scroller

    def _resolve_ai_quote_color(self, view: Gtk.TextView | None) -> Gdk.RGBA:
        fallback = Gdk.RGBA()
        fallback.parse("#ffffff")
        if not view:
            return fallback
        context = view.get_style_context()
        try:
            base = context.get_color()
        except TypeError:
            base = context.get_color(Gtk.StateFlags.NORMAL)
        quote = Gdk.RGBA()
        quote.red = base.red
        quote.green = base.green
        quote.blue = base.blue
        quote.alpha = DEFAULT_QUOTED_PHRASE_ALPHA
        return quote

    def _apply_link_spans(
        self,
        text: str,
        buffer: Gtk.TextBuffer | None,
        link_tags: list[Gtk.TextTag],
        link_lookup: dict[Gtk.TextTag, str],
        scroller: Gtk.ScrolledWindow | None,
    ) -> None:
        if not buffer:
            return
        table = buffer.get_tag_table()
        if table is None:
            return
        for tag in link_tags:
            try:
                table.remove(tag)
            except TypeError:
                pass
        link_tags.clear()
        link_lookup.clear()

        rendered_text, spans = self._extract_ai_link_spans(text)
        buffer.set_text(rendered_text)

        quote_color = self._resolve_ai_quote_color(
            self._summary_view if buffer is self._summary_buffer else None
        )
        if buffer is not self._summary_buffer:
            for state in self._ai_outputs.values():
                if state.buffer is buffer:
                    quote_color = self._resolve_ai_quote_color(state.view)
                    break
        for start, end, phrase in spans:
            if end <= start:
                continue
            start_iter = buffer.get_iter_at_offset(start)
            end_iter = buffer.get_iter_at_offset(end)
            tag = buffer.create_tag(
                None,
                foreground_rgba=quote_color,
                underline=Pango.Underline.NONE,
            )
            link_lookup[tag] = phrase
            buffer.apply_tag(tag, start_iter, end_iter)
            link_tags.append(tag)
        if scroller:
            scroller.queue_resize()

    def _apply_ai_output_links(self, text: str, state: AiOutputView) -> None:
        self._apply_link_spans(text, state.buffer, state.link_tags, state.link_lookup, state.scroller)

    def _apply_summary_links(self, text: str) -> None:
        self._apply_link_spans(
            text,
            self._summary_buffer,
            self._summary_link_tags,
            self._summary_link_tag_lookup,
            self._summary_scroller,
        )

    def _refresh_ai_quote_colors(self) -> None:
        if self._summary_view and self._summary_raw:
            self._apply_summary_links(self._summary_raw)
        for state in self._ai_outputs.values():
            if state.raw:
                self._apply_ai_output_links(state.raw, state)

    def _on_color_scheme_changed(self, *_args: object) -> None:
        self._refresh_ai_quote_colors()

    def _on_ai_output_view_mapped(self, _view: Gtk.TextView, view_name: str) -> None:
        state = self._ai_outputs.get(view_name)
        if not state or not state.raw:
            return
        self._apply_ai_output_links(state.raw, state)

    def _on_summary_view_mapped(self, _view: Gtk.TextView) -> None:
        if not self._summary_raw:
            return
        self._apply_summary_links(self._summary_raw)

    def _extract_ai_link_spans(self, text: str) -> tuple[str, list[tuple[int, int, str]]]:
        spans: list[tuple[int, int, str]] = []
        parts: list[str] = []
        cursor = 0
        offset = 0
        for match in AI_LINK_SPAN_RE.finditer(text):
            start, end = match.span()
            before = text[cursor:start]
            parts.append(before)
            offset += len(before)
            phrase = (match.group(1) or match.group(2) or "").strip()
            if phrase:
                link_phrase, trailing = split_link_phrase(phrase)
                if link_phrase:
                    parts.append(link_phrase)
                    spans.append((offset, offset + len(link_phrase), link_phrase))
                    offset += len(link_phrase)
                if trailing:
                    parts.append(trailing)
                    offset += len(trailing)
            cursor = end
        parts.append(text[cursor:])
        return "".join(parts), spans

    def _install_textview_link_controllers(self) -> None:
        if not self.textview:
            return
        if not self._textview_motion_controller:
            motion = Gtk.EventControllerMotion()
            motion.connect("motion", self._on_textview_motion)
            motion.connect("enter", self._on_textview_motion)
            motion.connect("leave", self._on_textview_leave)
            self.textview.add_controller(motion)
            self._textview_motion_controller = motion
        if not self._textview_click_gesture:
            click = Gtk.GestureClick.new()
            click.set_button(Gdk.BUTTON_PRIMARY)
            click.connect("released", self._on_textview_click)
            self.textview.add_controller(click)
            self._textview_click_gesture = click
        if not self._textview_focus_controller:
            focus_controller = Gtk.EventControllerFocus()
            focus_controller.connect("enter", self._on_textview_focus_enter)
            focus_controller.connect("leave", self._on_textview_focus_leave)
            self.textview.add_controller(focus_controller)
            self._textview_focus_controller = focus_controller

    def _on_textview_focus_enter(self, _controller: Gtk.EventControllerFocus) -> None:
        if self.textview:
            self.textview.set_cursor_visible(False)

    def _on_textview_focus_leave(self, _controller: Gtk.EventControllerFocus) -> None:
        if self.textview:
            self.textview.set_cursor_visible(False)

    def _on_textview_motion(
        self,
        _controller: Gtk.EventControllerMotion,
        x: float,
        y: float,
    ) -> None:
        if not self.textview:
            return
        link = self._link_at_coords(self.textview, x, y)
        if link:
            self.textview.set_cursor_from_name("pointer")
        else:
            self.textview.set_cursor_from_name(None)

    def _on_textview_leave(self, _controller: Gtk.EventControllerMotion) -> None:
        if self.textview:
            self.textview.set_cursor_from_name(None)

    def _on_textview_click(
        self,
        gesture: Gtk.GestureClick,
        _n_press: int,
        x: float,
        y: float,
    ) -> None:
        button = gesture.get_current_button()
        if button and button != Gdk.BUTTON_PRIMARY:
            return
        if not self.textview:
            return
        self.textview.grab_focus()
        self.textview.set_cursor_visible(False)
        link = self._link_at_coords(self.textview, x, y)
        if link is None:
            return
        kind, value = link
        if kind == "page":
            self._show_page_from_link(value)

    def _install_ai_output_link_controllers(self, state: AiOutputView) -> None:
        view = state.view
        if not view:
            return
        if not state.motion_controller:
            motion = Gtk.EventControllerMotion()
            motion.connect("motion", self._on_ai_output_motion, view, state.link_lookup)
            motion.connect("enter", self._on_ai_output_motion, view, state.link_lookup)
            motion.connect("leave", self._on_ai_output_leave, view)
            view.add_controller(motion)
            state.motion_controller = motion
        if not state.click_gesture:
            click = Gtk.GestureClick.new()
            click.set_button(Gdk.BUTTON_PRIMARY)
            click.connect("released", self._on_ai_output_click, view, state.link_lookup)
            view.add_controller(click)
            state.click_gesture = click
        if not state.focus_controller:
            focus_controller = Gtk.EventControllerFocus()
            focus_controller.connect("enter", self._ai_output_focus_enter, view)
            focus_controller.connect("leave", self._ai_output_focus_leave, view)
            view.add_controller(focus_controller)
            state.focus_controller = focus_controller

    def _ai_output_focus_enter(self, _controller: Gtk.EventControllerFocus, view: Gtk.TextView) -> None:
        view.set_cursor_visible(False)

    def _ai_output_focus_leave(self, _controller: Gtk.EventControllerFocus, view: Gtk.TextView) -> None:
        view.set_cursor_visible(False)

    def _ai_link_at_coords(
        self,
        textview: Gtk.TextView,
        x: float,
        y: float,
        lookup: dict[Gtk.TextTag, str],
    ) -> str | None:
        bx, by = textview.window_to_buffer_coords(Gtk.TextWindowType.WIDGET, int(x), int(y))
        iter_result = textview.get_iter_at_location(int(bx), int(by))
        if isinstance(iter_result, tuple):
            success, iter_ = iter_result
            if not success:
                return None
        else:
            iter_ = iter_result
        if iter_ is None:
            return None
        for tag in iter_.get_tags():
            link = lookup.get(tag)
            if link is not None:
                return link
        return None

    def _on_ai_output_motion(
        self,
        _controller: Gtk.EventControllerMotion,
        x: float,
        y: float,
        view: Gtk.TextView,
        lookup: dict[Gtk.TextTag, str],
    ) -> None:
        link = self._ai_link_at_coords(view, x, y, lookup)
        if link:
            view.set_cursor_from_name("pointer")
        else:
            view.set_cursor_from_name(None)

    def _on_ai_output_leave(self, _controller: Gtk.EventControllerMotion, view: Gtk.TextView) -> None:
        view.set_cursor_from_name(None)

    def _on_ai_output_click(
        self,
        gesture: Gtk.GestureClick,
        _n_press: int,
        x: float,
        y: float,
        view: Gtk.TextView,
        lookup: dict[Gtk.TextTag, str],
    ) -> None:
        button = gesture.get_current_button()
        if button and button != Gdk.BUTTON_PRIMARY:
            return
        view.grab_focus()
        view.set_cursor_visible(False)
        phrase = self._ai_link_at_coords(view, x, y, lookup)
        if not phrase:
            return
        self._activate_ai_link(phrase)

    def _install_summary_link_controllers(self) -> None:
        if not self._summary_view:
            return
        if not self._summary_motion_controller:
            motion = Gtk.EventControllerMotion()
            motion.connect("motion", self._on_summary_motion)
            motion.connect("enter", self._on_summary_motion)
            motion.connect("leave", self._on_summary_leave)
            self._summary_view.add_controller(motion)
            self._summary_motion_controller = motion
        if not self._summary_click_gesture:
            click = Gtk.GestureClick.new()
            click.set_button(Gdk.BUTTON_PRIMARY)
            click.connect("released", self._on_summary_click)
            self._summary_view.add_controller(click)
            self._summary_click_gesture = click
        if not self._summary_focus_controller:
            focus_controller = Gtk.EventControllerFocus()
            focus_controller.connect("enter", self._summary_focus_enter)
            focus_controller.connect("leave", self._summary_focus_leave)
            self._summary_view.add_controller(focus_controller)
            self._summary_focus_controller = focus_controller

    def _summary_focus_enter(self, _controller: Gtk.EventControllerFocus) -> None:
        if self._summary_view:
            self._summary_view.set_cursor_visible(False)

    def _summary_focus_leave(self, _controller: Gtk.EventControllerFocus) -> None:
        if self._summary_view:
            self._summary_view.set_cursor_visible(False)

    def _on_summary_search_changed(self, entry: Gtk.SearchEntry) -> None:
        query = entry.get_text().strip()
        if query == self._summary_search_query:
            return
        self._summary_search_query = query
        self._refresh_summary_search(reset_active=True)

    def _on_summary_search_activate(self, entry: Gtk.SearchEntry) -> None:
        query = entry.get_text().strip()
        if query != self._summary_search_query:
            self._summary_search_query = query
            self._refresh_summary_search(reset_active=True)
        if not self._summary_search_matches:
            if query:
                self._transient_toast("No matches found in the file.")
            return
        if self._summary_search_index < 0:
            self._summary_search_index = 0
        else:
            self._summary_search_index = (self._summary_search_index + 1) % len(self._summary_search_matches)
        self._apply_summary_search_highlights()
        self._scroll_to_summary_match(self._summary_search_index)

    def _refresh_summary_search(self, *, reset_active: bool = False) -> None:
        if reset_active:
            self._summary_search_index = -1
        self._update_summary_search_matches()

    def _ensure_summary_search_tags(self) -> None:
        if not self._summary_buffer:
            return
        table = self._summary_buffer.get_tag_table()
        if table is None:
            return
        if self._summary_search_tag is None:
            tag = table.lookup("summary-search-match")
            if tag is None:
                tag = self._summary_buffer.create_tag(
                    "summary-search-match",
                    background="#f7dcc3",
                    foreground="#3f2b1a",
                )
            self._summary_search_tag = tag
        if self._summary_search_current_tag is None:
            tag = table.lookup("summary-search-current")
            if tag is None:
                tag = self._summary_buffer.create_tag(
                    "summary-search-current",
                    background=DEFAULT_MATCH_COLOR,
                    foreground="#2b1600",
                )
            self._summary_search_current_tag = tag

    def _clear_summary_search_tags(self) -> None:
        if not self._summary_buffer:
            return
        start = self._summary_buffer.get_start_iter()
        end = self._summary_buffer.get_end_iter()
        if self._summary_search_tag:
            self._summary_buffer.remove_tag(self._summary_search_tag, start, end)
        if self._summary_search_current_tag:
            self._summary_buffer.remove_tag(self._summary_search_current_tag, start, end)

    def _update_summary_search_matches(self) -> None:
        if not self._summary_buffer:
            return
        self._summary_search_matches = []
        self._clear_summary_search_tags()
        query = self._summary_search_query
        if not query:
            return
        start = self._summary_buffer.get_start_iter()
        end = self._summary_buffer.get_end_iter()
        flags = Gtk.TextSearchFlags.CASE_INSENSITIVE
        while True:
            result = start.forward_search(query, flags, end)
            if result is None:
                break
            match_start, match_end = result
            self._summary_search_matches.append(
                (match_start.get_offset(), match_end.get_offset())
            )
            start = match_end
        self._apply_summary_search_highlights()

    def _apply_summary_search_highlights(self) -> None:
        if not self._summary_buffer:
            return
        self._ensure_summary_search_tags()
        if not self._summary_search_tag or not self._summary_search_current_tag:
            return
        start = self._summary_buffer.get_start_iter()
        end = self._summary_buffer.get_end_iter()
        self._summary_buffer.remove_tag(self._summary_search_tag, start, end)
        self._summary_buffer.remove_tag(self._summary_search_current_tag, start, end)
        for start_offset, end_offset in self._summary_search_matches:
            start_iter = self._summary_buffer.get_iter_at_offset(start_offset)
            end_iter = self._summary_buffer.get_iter_at_offset(end_offset)
            self._summary_buffer.apply_tag(self._summary_search_tag, start_iter, end_iter)
        if 0 <= self._summary_search_index < len(self._summary_search_matches):
            start_offset, end_offset = self._summary_search_matches[self._summary_search_index]
            start_iter = self._summary_buffer.get_iter_at_offset(start_offset)
            end_iter = self._summary_buffer.get_iter_at_offset(end_offset)
            self._summary_buffer.apply_tag(self._summary_search_current_tag, start_iter, end_iter)

    def _scroll_to_summary_match(self, index: int) -> None:
        if not self._summary_view or not self._summary_buffer:
            return
        if index < 0 or index >= len(self._summary_search_matches):
            return
        start_offset, _ = self._summary_search_matches[index]
        start_iter = self._summary_buffer.get_iter_at_offset(start_offset)
        self._summary_view.scroll_to_iter(start_iter, 0.15, True, 0.1, 0.1)

    def _connect_summary_scroll_watch(self) -> None:
        if not self._summary_scroller:
            return
        vadj = self._summary_scroller.get_vadjustment()
        if not vadj:
            return
        if self._summary_scroll_handler_id is not None:
            try:
                vadj.disconnect(self._summary_scroll_handler_id)
            except (TypeError, RuntimeError):
                pass
        self._summary_scroll_handler_id = vadj.connect("value-changed", self._on_summary_scroll)

    def _summary_scroll_key(self, path: Path) -> str:
        return str(path.expanduser().resolve(strict=False))

    def _summary_scroll_fraction(self, adjustment: Gtk.Adjustment) -> float:
        lower = adjustment.get_lower()
        upper = adjustment.get_upper()
        page_size = adjustment.get_page_size()
        total = upper - lower - page_size
        if total <= 0:
            return 0.0
        value = adjustment.get_value()
        return (value - lower) / total

    def _on_summary_scroll(self, adjustment: Gtk.Adjustment) -> None:
        if self._summary_scroll_restore_guard:
            return
        path = self._summary_loaded_path
        if not path:
            return
        fraction = self._summary_scroll_fraction(adjustment)
        fraction = min(1.0, max(0.0, fraction))
        key = self._summary_scroll_key(path)
        if abs(self._summary_scroll_positions.get(key, -1.0) - fraction) < 0.001:
            return
        self._summary_scroll_positions[key] = fraction
        self._schedule_summary_scroll_save()

    def _schedule_summary_scroll_save(self) -> None:
        if self._summary_scroll_save_source_id is not None:
            return
        self._summary_scroll_save_source_id = GLib.timeout_add(
            500, self._flush_summary_scroll_positions
        )

    def _flush_summary_scroll_positions(self) -> bool:
        self._summary_scroll_save_source_id = None
        save_summary_read_positions(self._summary_scroll_positions)
        return False

    def _restore_summary_scroll_position(self, path: Path | None) -> None:
        if not path or not self._summary_scroller:
            return
        key = self._summary_scroll_key(path)
        fraction = self._summary_scroll_positions.get(key)
        if fraction is None:
            return

        def _apply() -> bool:
            if not self._summary_scroller:
                return False
            vadj = self._summary_scroller.get_vadjustment()
            if not vadj:
                return False
            lower = vadj.get_lower()
            upper = vadj.get_upper()
            page_size = vadj.get_page_size()
            total = upper - lower - page_size
            if total <= 0:
                return False
            target = lower + min(1.0, max(0.0, fraction)) * total
            self._summary_scroll_restore_guard = True
            vadj.set_value(target)
            self._summary_scroll_restore_guard = False
            return False

        GLib.idle_add(_apply)

    def _summary_link_at_coords(self, textview: Gtk.TextView, x: float, y: float) -> str | None:
        bx, by = textview.window_to_buffer_coords(Gtk.TextWindowType.WIDGET, int(x), int(y))
        iter_result = textview.get_iter_at_location(int(bx), int(by))
        if isinstance(iter_result, tuple):
            success, iter_ = iter_result
            if not success:
                return None
        else:
            iter_ = iter_result
        if iter_ is None:
            return None
        for tag in iter_.get_tags():
            link = self._summary_link_tag_lookup.get(tag)
            if link is not None:
                return link
        return None

    def _on_summary_motion(
        self,
        _controller: Gtk.EventControllerMotion,
        x: float,
        y: float,
    ) -> None:
        if not self._summary_view:
            return
        link = self._summary_link_at_coords(self._summary_view, x, y)
        if link:
            self._summary_view.set_cursor_from_name("pointer")
        else:
            self._summary_view.set_cursor_from_name(None)

    def _on_summary_leave(self, _controller: Gtk.EventControllerMotion) -> None:
        if self._summary_view:
            self._summary_view.set_cursor_from_name(None)

    def _on_summary_click(
        self,
        gesture: Gtk.GestureClick,
        _n_press: int,
        x: float,
        y: float,
    ) -> None:
        button = gesture.get_current_button()
        if button and button != Gdk.BUTTON_PRIMARY:
            return
        if not self._summary_view:
            return
        self._summary_view.grab_focus()
        self._summary_view.set_cursor_visible(False)
        phrase = self._summary_link_at_coords(self._summary_view, x, y)
        if not phrase:
            return
        self._activate_ai_link(phrase)

    def _activate_ai_link(self, phrase: str) -> None:
        cleaned = phrase.strip()
        if self._grep_entry:
            self._grep_entry.set_text(cleaned)
        self._apply_grep(cleaned)

    def _show_page_from_link(self, page_str: str) -> None:
        try:
            page_num = int(page_str)
        except ValueError:
            return
        if not self.pages:
            return
        self._deactivate_continuous_view(reload=False)
        idx = bisect.bisect_left(self.pages, page_num)
        if idx >= len(self.pages) or self.pages[idx] != page_num:
            self._transient_toast(f"Page {page_num:04d} not available")
            return
        self.current_index = idx
        if self._showing_grep_results:
            self._showing_grep_results = False
        self._load_current()

    def _load_image_for_page(self, page: int, *, silent: bool = False) -> bool:
        if not (self._image_picture and self._image_scroller):
            return False
        text_path = self.page_to_path.get(page)
        if not text_path:
            if not silent:
                self._transient_toast(f"Text for page {page:04d} not available")
            return False
        image_dir = self.images_dir
        image_path = image_dir / f"{page:04d}.png"
        if not image_path.exists():
            if not silent:
                self._transient_toast(f"Image {image_path.name} not found")
            return False

        picture_file = Gio.File.new_for_path(str(image_path))
        self._image_picture.set_file(picture_file)
        self._image_picture.set_alternative_text(f"Page {page:04d} image")

        vadj = self._image_scroller.get_vadjustment()
        if vadj:
            GLib.idle_add(vadj.set_value, vadj.get_lower())
        hadj = self._image_scroller.get_hadjustment()
        if hadj:
            GLib.idle_add(hadj.set_value, hadj.get_lower())
        return True

    def _clear_image_view(self) -> None:
        if self._image_picture:
            self._image_picture.set_file(None)
            self._image_picture.set_alternative_text("")

    def _show_image_update_visible(self) -> None:
        if not self._content_stack:
            return
        target = "image" if self._show_image else "text"
        current = self._content_stack.get_visible_child_name()
        if current != target:
            self._content_stack.set_visible_child_name(target)

    def _sync_show_image_action(self) -> None:
        if not self._show_image_action:
            self._update_show_image_toggle_button()
            return
        state = self._show_image_action.get_state()
        current = state.get_boolean() if state is not None else None
        if current != self._show_image:
            self._show_image_action.set_state(GLib.Variant.new_boolean(self._show_image))
        self._update_show_image_toggle_button()

    def _update_show_image_toggle_button(self) -> None:
        if not self._show_image_button or not self._show_image_icon:
            return
        self._show_image_button_guard = True
        try:
            self._show_image_button.set_active(self._show_image)
        finally:
            self._show_image_button_guard = False
        icon_name = self._image_icon_name_on if self._show_image else self._image_icon_name_off
        self._show_image_icon.set_from_icon_name(icon_name)
        tooltip = "Disable image view (Ctrl+I)" if self._show_image else "Enable image view (Ctrl+I)"
        self._show_image_button.set_tooltip_text(tooltip)

    def _update_summary_buttons(self) -> None:
        if self._choose_summary_button:
            choose_hint = "Pick a text or markdown summary file to view"
            if self._summary_file_path:
                if self._summary_file_path.exists():
                    choose_hint = f"Change summary file (current: {self._summary_file_path.name})"
                else:
                    choose_hint = (
                        f"Saved summary missing, choose a new file (expected {self._summary_file_path.name})"
                    )
            self._choose_summary_button.set_tooltip_text(choose_hint)

    def _set_font_preferences(self, *, font_size_pt: int | None = None, ai_font_size_pt: int | None = None) -> None:
        base = self._font_size_pt
        ai = self._ai_font_size_pt
        if font_size_pt is not None:
            base = _coerce_font_size(font_size_pt, DEFAULT_FONT_SIZE_PT)
        if ai_font_size_pt is not None:
            ai = _coerce_font_size(ai_font_size_pt, max(base, DEFAULT_AI_FONT_SIZE_PT))
        self._font_size_pt = base
        self._ai_font_size_pt = ai
        save_font_preferences(base, ai)
        self._apply_text_color(self._current_text_color)

    def get_font_preferences(self) -> tuple[int, int]:
        return self._font_size_pt, self._ai_font_size_pt

    def update_font_sizes(self, *, font_size_pt: int | None = None, ai_font_size_pt: int | None = None) -> None:
        self._set_font_preferences(font_size_pt=font_size_pt, ai_font_size_pt=ai_font_size_pt)

    def update_ai_font_size(self, ai_font_size_pt: int) -> None:
        self.update_font_sizes(ai_font_size_pt=ai_font_size_pt)

    def _summary_has_text(self) -> bool:
        return bool(self._summary_buffer and self._summary_buffer.get_char_count() > 0)

    def _show_summary_view(self, *, switch_view: bool = True) -> None:
        if switch_view:
            self._set_ai_view(AI_VIEW_FILE)
        if self._summary_scroller:
            self._summary_scroller.queue_resize()
        if self._summary_view:
            self._summary_view.set_cursor_visible(False)

    def _set_show_image(self, enabled: bool, *, silent: bool = False) -> bool:
        if enabled:
            if self._continuous_view:
                if not silent:
                    self._transient_toast("Show Image is unavailable in continuous view.")
                self._sync_show_image_action()
                return False
            if self._showing_grep_results:
                if not silent:
                    self._transient_toast("Show Image is unavailable while multiple pages are displayed.")
                self._sync_show_image_action()
                return False
            if not self.pages:
                if not silent:
                    self._transient_toast("No page available to display an image.")
                self._sync_show_image_action()
                return False
            page = self.pages[self.current_index]
            if not self._load_image_for_page(page, silent=silent):
                self._clear_image_view()
                self._show_image = False
                self._show_image_update_visible()
                self._sync_show_image_action()
                return False
            self._show_image = True
            self._show_image_update_visible()
            self._sync_show_image_action()
            return True

        # Always force the stack back to the text view when disabling image mode,
        # even if the flag was already false (e.g., after returning from a different view).
        self._show_image = False
        self._show_image_update_visible()
        self._clear_image_view()
        self._sync_show_image_action()
        return True

    def _link_at_coords(self, textview: Gtk.TextView, x: float, y: float) -> tuple[str, str] | None:
        bx, by = textview.window_to_buffer_coords(Gtk.TextWindowType.WIDGET, int(x), int(y))
        iter_result = textview.get_iter_at_location(int(bx), int(by))
        if isinstance(iter_result, tuple):
            success, iter_ = iter_result
            if not success:
                return None
        else:
            iter_ = iter_result
        if iter_ is None:
            return None
        for tag in iter_.get_tags():
            link = self._link_tag_lookup.get(tag)
            if link is not None:
                return link
        return None

    def _update_page_nav_buttons(self) -> None:
        enabled = bool(self.pages) and not self._continuous_view and not self._showing_grep_results
        if self._page_back_ten_button:
            self._page_back_ten_button.set_sensitive(enabled)
        if self._page_back_one_button:
            self._page_back_one_button.set_sensitive(enabled)
        if self._page_forward_one_button:
            self._page_forward_one_button.set_sensitive(enabled)
        if self._page_forward_ten_button:
            self._page_forward_ten_button.set_sensitive(enabled)
        if self._page_status_label:
            if self.pages and 0 <= self.current_index < len(self.pages):
                current_page = self.pages[self.current_index]
                total_pages = len(self.pages)
                self._page_status_label.set_text(f"{current_page}/{total_pages}")
            else:
                self._page_status_label.set_text("--/--")

    def _update_header(self) -> None:
        self._update_page_nav_buttons()
        if self._showing_grep_results:
            summary = f"Grep results ({len(self._matching_pages)} pages)"
            self._set_window_title(summary, summary)
            return
        if self._continuous_view:
            if not self.pages:
                self._set_window_title(None, "Continuous view")
                return
            page = self.pages[self.current_index]
            summary = f"Continuous view - starting at {page:04d}"
            self._set_window_title(None, summary)
            return
        if not self.pages:
            self._set_window_title("No pages found", "No pages found")
            return
        page = self.pages[self.current_index]
        self._set_window_title(None, f"Page {page:04d}")

    def _read_page_text(self, page: int) -> tuple[str, str, list[int]]:
        if (
            page in self._page_cache
            and page in self._page_search_cache
            and page in self._page_search_map_cache
        ):
            return (
                self._page_cache[page],
                self._page_search_cache[page],
                self._page_search_map_cache[page],
            )
        path = self.page_to_path.get(page)
        if not path:
            return "", "", []
        try:
            with io.open(path, "r", encoding="utf-8", errors="replace") as handle:
                content = handle.read()
        except Exception as exc:  # noqa: BLE001
            content = f"Error reading {path.name}: {exc}"
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        normalized, norm_to_orig = normalize_text_for_search_with_map(content)

        self._page_cache[page] = content
        self._page_search_cache[page] = normalized
        self._page_search_map_cache[page] = norm_to_orig
        return content, normalized, norm_to_orig

    def _render_page_display(
        self,
        page: int,
        content: str,
        highlights: list[tuple[int, int]] | None,
    ) -> tuple[str, list[tuple[int, int]] | None]:
        header = f"{page:04d}\n\n"
        adjusted: list[tuple[int, int]] = []
        if highlights:
            offset = len(header)
            for start, end in highlights:
                if end <= start:
                    continue
                adjusted.append((start + offset, end + offset))
        return header + content, adjusted if adjusted else None

    def _load_current(self) -> None:
        if self._showing_grep_results and self._grep_combined_text:
            self._set_show_image(False, silent=True)
            self._show_grep_results()
            return
        if not self.pages:
            self._set_show_image(False, silent=True)
            return
        page = self.pages[self.current_index]
        path = self.page_to_path.get(page)
        if not path or not path.exists():
            display_text, highlight_spans = self._render_page_display(
                page, f"Missing file for page {page:04d}", None
            )
            self._set_text(display_text, highlight_spans)
            self._set_show_image(False, silent=True)
            self._update_header()
            return
        content, _, _ = self._read_page_text(page)
        highlights = self._grep_hits.get(page)
        display_text, highlight_spans = self._render_page_display(page, content, highlights)
        self._set_text(display_text, highlight_spans)
        if self._show_image:
            if not self._load_image_for_page(page):
                self._set_show_image(False, silent=True)
            else:
                self._show_image_update_visible()
        else:
            self._show_image_update_visible()
        self._update_header()
        self._sync_sidebar_active_page()

    def _clear_grep_state(self) -> None:
        self._grep_regex = None
        self._grep_hits.clear()
        self._matching_pages.clear()
        self._matching_lookup.clear()
        self._grep_combined_text = None
        self._grep_combined_highlights = []
        self._showing_grep_results = False

    def _apply_grep(self, phrase: str) -> None:
        phrase = phrase.strip()
        self._deactivate_continuous_view(reload=False)
        if phrase:
            self._set_show_image(False, silent=True)
        if not phrase:
            self._grep_phrase_raw = None
            self._clear_grep_state()
            self._load_current()
            return

        self._grep_phrase_raw = phrase
        try:
            self._prepare_grep()
        except re.error as exc:
            self._transient_toast(f"Invalid grep pattern: {exc}")
            return

        if not self._matching_pages:
            self._showing_grep_results = False
            self._transient_toast("No pages matched the grep phrase")
            self._load_current()
            return

        first_page = self._matching_pages[0]
        if first_page in self.pages:
            self.current_index = self.pages.index(first_page)

        if len(self._matching_pages) == 1:
            self._showing_grep_results = False
            self._grep_combined_text = None
            self._grep_combined_highlights = []
            self._load_current()
            return

        self._showing_grep_results = True
        self._show_grep_results()

    def _map_normalized_span_to_original(
        self,
        norm_to_orig: list[int],
        start: int,
        end: int,
        original_len: int,
    ) -> tuple[int, int] | None:
        if not norm_to_orig or end <= start or end <= 0:
            return None
        max_index = len(norm_to_orig) - 1
        if max_index < 0:
            return None
        if start < 0:
            start = 0
        if start > max_index:
            return None
        if end > len(norm_to_orig):
            end = len(norm_to_orig)
        if end <= start:
            return None
        start_orig = norm_to_orig[start]
        end_orig = norm_to_orig[end - 1] + 1
        if end_orig <= start_orig or start_orig >= original_len:
            return None
        if end_orig > original_len:
            end_orig = original_len
        return start_orig, end_orig

    def _prepare_grep(self) -> None:
        assert self._grep_phrase_raw is not None
        phrase = preprocess_phrase(self._grep_phrase_raw)
        self._clear_grep_state()
        self._grep_regex = re.compile(build_pattern(phrase, MAX_BREAKS), re.IGNORECASE | re.DOTALL)

        for page in self.pages:
            content, normalized, norm_to_orig = self._read_page_text(page)
            if not normalized or not self._grep_regex:
                continue
            matches = list(self._grep_regex.finditer(normalized))
            if matches:
                original_hits: list[tuple[int, int]] = []
                original_len = len(content)
                for match in matches:
                    mapped = self._map_normalized_span_to_original(
                        norm_to_orig,
                        match.start(),
                        match.end(),
                        original_len,
                    )
                    if mapped:
                        original_hits.append(mapped)
                if original_hits:
                    self._grep_hits[page] = original_hits

        if self._grep_hits:
            self._matching_pages.extend(sorted(self._grep_hits.keys()))
            self._matching_lookup.update({page: idx for idx, page in enumerate(self._matching_pages)})
            self._build_grep_document()

    def _build_grep_document(self) -> None:
        parts: list[str] = []
        highlights: list[tuple[int, int]] = []
        offset = 0
        for page in self._matching_pages:
            content, _, _ = self._read_page_text(page)
            header = f"{page:04d}\n\n"
            parts.append(header)
            parts.append(content)
            parts.append("\n\n")
            header_len = len(header)
            for start, end in self._grep_hits.get(page, []):
                highlights.append((offset + header_len + start, offset + header_len + end))
            offset += header_len + len(content) + 2
        self._grep_combined_text = "".join(parts) if parts else None
        self._grep_combined_highlights = highlights
        self._showing_grep_results = bool(parts)

    def _show_grep_results(self) -> None:
        if not self._showing_grep_results or not self._grep_combined_text:
            return
        if self._grep_entry and self._grep_phrase_raw is not None:
            self._grep_entry.set_text(self._grep_phrase_raw)
        self._set_text(self._grep_combined_text, self._grep_combined_highlights)
        self._update_header()
        self._sync_sidebar_active_page()

    def _install_navigation_controllers(self) -> None:
        if not self.win:
            return
        def attach_scroll_controller(widget: Gtk.Widget | None) -> bool:
            if widget is None:
                return False
            controller = Gtk.EventControllerScroll.new(Gtk.EventControllerScrollFlags.VERTICAL)
            controller.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
            controller.connect("scroll", self._on_scroll)
            widget.add_controller(controller)
            return True

        attached = attach_scroll_controller(self.scroller)
        if not attached:
            attach_scroll_controller(self.win)
        attach_scroll_controller(self._image_scroller)

        key_ctrl = Gtk.EventControllerKey.new()
        key_ctrl.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        key_ctrl.connect("key-pressed", self._on_key)
        self.win.add_controller(key_ctrl)

    def _install_actions(self) -> None:
        choose_input = Gio.SimpleAction.new("choose_input", None)
        choose_input.connect("activate", self._on_choose_input_dir)
        self.add_action(choose_input)

        open_toc = Gio.SimpleAction.new("open_toc", None)
        open_toc.connect("activate", self._on_open_toc_window)
        self.add_action(open_toc)

        open_ai_settings = Gio.SimpleAction.new("open_ai_settings", None)
        open_ai_settings.connect("activate", self._on_open_ai_settings)
        self.add_action(open_ai_settings)

        toggle_sidebar = Gio.SimpleAction.new_stateful(
            "toggle_toc_sidebar",
            None,
            GLib.Variant.new_boolean(self._toc_sidebar_visible),
        )
        toggle_sidebar.connect("change-state", self._on_toggle_toc_sidebar)
        self.add_action(toggle_sidebar)
        self._toc_sidebar_action = toggle_sidebar

        continuous_action = Gio.SimpleAction.new_stateful(
            "toggle_continuous_view",
            None,
            GLib.Variant.new_boolean(self._continuous_view),
        )
        continuous_action.connect("change-state", self._on_toggle_continuous_view)
        self.add_action(continuous_action)
        self._continuous_action = continuous_action

        show_image_action = Gio.SimpleAction.new_stateful(
            "toggle_show_image",
            None,
            GLib.Variant.new_boolean(self._show_image),
        )
        show_image_action.connect("change-state", self._on_toggle_show_image)
        self.add_action(show_image_action)
        self._show_image_action = show_image_action

        focus_rag_question = Gio.SimpleAction.new("focus_rag_question", None)
        focus_rag_question.connect("activate", lambda _a, _p: self._focus_rag_question_entry())
        self.add_action(focus_rag_question)

        for name, cb in {
            "next": self._go_next,
            "prev": self._go_prev,
            "first": self._go_first,
            "last": self._go_last,
        }.items():
            act = Gio.SimpleAction.new(name, None)
            act.connect("activate", lambda a, p, cb=cb: cb())  # noqa: ARG005
            self.add_action(act)

        self.set_accels_for_action("app.prev", ["Up"])
        self.set_accels_for_action("app.next", ["Down"])
        self.set_accels_for_action("app.first", ["Home"])
        self.set_accels_for_action("app.last", ["End"])
        self.set_accels_for_action("app.toggle_toc_sidebar", ["<Primary><Shift>z"])
        self.set_accels_for_action("app.toggle_continuous_view", ["<Primary><Shift>c"])
        self.set_accels_for_action("app.toggle_show_image", ["<Primary>i"])
        self.set_accels_for_action("app.focus_rag_question", ["<Primary>q"])
        self._set_sidebar_visible(self._toc_sidebar_visible)

    def _on_choose_input_dir(self, _action: Gio.SimpleAction, _param: GLib.Variant | None) -> None:
        if not self.win:
            return
        dialog = Gtk.FileDialog()
        dialog.set_title("Select Input Directory")
        dialog.set_modal(True)
        if self.input_dir.exists():
            try:
                dialog.set_initial_folder(Gio.File.new_for_path(str(self.input_dir)))
            except (TypeError, AttributeError):
                pass
        dialog.select_folder(self.win, None, self._on_input_dir_dialog_response)
        self._input_dir_dialog = dialog

    def _on_input_dir_dialog_response(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        try:
            file = dialog.select_folder_finish(result)
        except GLib.Error as exc:
            if not exc.matches(Gio.io_error_quark(), Gio.IOErrorEnum.CANCELLED):
                self._transient_toast(f"Directory selection failed: {exc.message}")
        else:
            path_str = file.get_path() if file else None
            if path_str:
                self._apply_input_dir(Path(path_str))
        if self._input_dir_dialog is dialog:
            self._input_dir_dialog = None

    def _on_open_toc_window(self, _action: Gio.SimpleAction, _param: GLib.Variant | None) -> None:
        window = self._ensure_toc_window()
        if window:
            window.present()

    def _on_open_ai_settings(self, _action: Gio.SimpleAction, _param: GLib.Variant | None) -> None:
        try:
            window = self._ensure_ai_settings_window()
        except Exception as exc:  # noqa: BLE001
            self._transient_toast(f"Settings unavailable: {exc}")
            return
        if window:
            window.present()

    def _ensure_toc_window(self) -> TocWindow | None:
        if self._toc_window:
            return self._toc_window
        window = TocWindow(self)
        window.connect("close-request", self._on_toc_window_close_request)
        self._toc_window = window
        return window

    def _ensure_ai_settings_window(self) -> "AiSettingsWindow" | None:
        if self._ai_settings_window:
            return self._ai_settings_window
        window = AiSettingsWindow(self)
        window.connect("close-request", self._on_ai_settings_window_close_request)
        self._ai_settings_window = window
        return window

    def _on_toc_window_close_request(self, window: TocWindow) -> bool:
        if self._toc_window is window:
            self._toc_window = None
        return False

    def _on_ai_settings_window_close_request(self, window: "AiSettingsWindow") -> bool:
        if self._ai_settings_window is window:
            self._ai_settings_window = None
        return False

    def _on_main_window_close_request(self, _window: Adw.ApplicationWindow) -> bool:
        if self._summary_scroll_save_source_id is not None:
            GLib.source_remove(self._summary_scroll_save_source_id)
            self._summary_scroll_save_source_id = None
        if self._summary_scroll_positions:
            save_summary_read_positions(self._summary_scroll_positions)
        return False

    def on_ai_settings_saved(self, settings: AiSettings) -> None:
        self._ai_settings = settings
        if not self._ai_settings.page_prompt.strip():
            self._ai_settings.page_prompt = DEFAULT_SUMMARIZATION_PROMPT
        if not self._ai_settings.range_prompt.strip():
            self._ai_settings.range_prompt = DEFAULT_SUMMARIZATION_PROMPT
        if not self._ai_settings.rag_prompt.strip():
            self._ai_settings.rag_prompt = DEFAULT_RAG_PROMPT
        self._kickoff_rag_background_load()
        if self.textview:
            self._load_current()
        self._transient_toast("AI settings updated.")

    def _apply_input_dir(self, path: Path) -> None:
        target = path.expanduser()
        resolved = target.resolve(strict=False)
        if not resolved.exists() or not resolved.is_dir():
            self._transient_toast(f"Directory not found: {resolved}")
            return
        normalized = _normalize_input_dir(resolved)
        if not normalized.exists() or not normalized.is_dir():
            self._transient_toast(f"Directory not found: {normalized}")
            return
        self._deactivate_continuous_view(reload=False)
        self._set_show_image(False, silent=True)
        self._reset_view_states()
        self.input_dir = normalized
        save_input_dir_to_config(normalized)
        if not self.text_dir.exists():
            self._transient_toast(f"'text_record' directory not found in: {normalized}")
        self._showing_grep_results = False
        self._grep_phrase_raw = None
        self._grep_regex = None
        self._grep_hits.clear()
        self._matching_pages.clear()
        self._matching_lookup.clear()
        self._grep_combined_text = None
        self._grep_combined_highlights = []
        self._scan_pages()
        self._load_toc_from_disk_async()
        self._kickoff_rag_background_load()
        if self.pages:
            self.current_index = 0
            self._load_current()
            self._persist_active_view_state()
        else:
            self._set_window_title("No pages found", "No pages found")
            self._set_text("No .txt pages found in:\n" + str(self.text_dir))
        if self._toc_window:
            self._toc_window.on_input_dir_changed()

    def open_regex_dir_dialog(self, parent: Gtk.Window) -> None:
        dialog = Gtk.FileDialog()
        dialog.set_title("Select Regex Directory")
        dialog.set_modal(True)
        if self.regex_dir.exists():
            try:
                dialog.set_initial_folder(Gio.File.new_for_path(str(self.regex_dir)))
            except (TypeError, AttributeError):
                pass
        dialog.select_folder(parent, None, self._on_regex_dir_dialog_response)
        self._regex_dir_dialog = dialog

    def _on_regex_dir_dialog_response(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        try:
            file = dialog.select_folder_finish(result)
        except GLib.Error as exc:
            if not exc.matches(Gio.io_error_quark(), Gio.IOErrorEnum.CANCELLED):
                self._transient_toast(f"Directory selection failed: {exc.message}")
        else:
            path_str = file.get_path() if file else None
            if path_str:
                self._apply_regex_dir(Path(path_str))
        if self._regex_dir_dialog is dialog:
            self._regex_dir_dialog = None

    def _apply_regex_dir(self, path: Path) -> None:
        target = path.expanduser()
        resolved = target.resolve(strict=False)
        if not resolved.exists() or not resolved.is_dir():
            self._transient_toast(f"Directory not found: {resolved}")
            return
        self.regex_dir = resolved
        save_regex_dir_to_config(resolved)
        if self._toc_window:
            self._toc_window.on_regex_dir_changed()

    def _on_scroll(self, ctrl: Gtk.EventControllerScroll, dx: float, dy: float) -> bool:
        state = ctrl.get_current_event_state()
        if state & Gdk.ModifierType.CONTROL_MASK:
            if dy > 0:
                self._go_next()
            elif dy < 0:
                self._go_prev()
            return True
        return False

    def _on_key(self, _ctrl: Gtk.EventControllerKey, keyval: int, keycode: int, state: int) -> bool:  # noqa: ARG002
        key = Gdk.keyval_name(keyval)
        if key == "Up":
            self._go_prev(); return True
        if key == "Down":
            self._go_next(); return True
        if key == "Home":
            self._go_first(); return True
        if key == "End":
            self._go_last(); return True
        if key == "f" and (state & Gdk.ModifierType.CONTROL_MASK):
            self._focus_grep_entry(); return True
        if key in ("A", "a") and (state & Gdk.ModifierType.CONTROL_MASK) and (state & Gdk.ModifierType.SHIFT_MASK):
            if self._ai_panel_toggle:
                current_visible = self._ai_panel_toggle.get_active()
            elif self._ai_panel_revealer:
                current_visible = self._ai_panel_revealer.get_child_revealed()
            else:
                current_visible = False
            new_visible = not bool(current_visible)
            self._set_ai_panel_visible(new_visible)
            if new_visible:
                self._focus_rag_question_entry()
            return True
        return False

    def _on_page_back_ten_clicked(self, _button: Gtk.Button) -> None:
        if self._continuous_view or self._showing_grep_results:
            return
        self._go_prev_ten()

    def _on_page_back_one_clicked(self, _button: Gtk.Button) -> None:
        if self._continuous_view or self._showing_grep_results:
            return
        self._go_prev()

    def _on_page_forward_one_clicked(self, _button: Gtk.Button) -> None:
        if self._continuous_view or self._showing_grep_results:
            return
        self._go_next()

    def _on_page_forward_ten_clicked(self, _button: Gtk.Button) -> None:
        if self._continuous_view or self._showing_grep_results:
            return
        self._go_next_ten()

    def _focus_grep_entry(self) -> None:
        if self._grep_entry:
            self._grep_entry.grab_focus()
            self._grep_entry.select_region(0, -1)

    def _focus_rag_question_entry(self) -> None:
        self._ensure_ai_panel_visible()
        self._set_ai_view(AI_VIEW_QA)
        if not self._rag_question_entry:
            return

        def _focus() -> bool:
            self._rag_question_entry.grab_focus()
            self._rag_question_entry.select_region(0, -1)
            return False

        GLib.idle_add(_focus)

    def _on_grep_entry_activate(self, entry: Gtk.Entry) -> None:
        phrase = entry.get_text()
        self._apply_grep(phrase)

    def _on_grep_search_clicked(self, _button: Gtk.Button) -> None:
        if not self._grep_entry:
            return
        self._apply_grep(self._grep_entry.get_text())

    def _on_grep_search_highlighted_clicked(self, _button: Gtk.Button) -> None:
        phrase = self._get_main_text_selection()
        if not phrase:
            phrase = self._get_ai_panel_selection()
        if not phrase:
            self._transient_toast("Highlight text in the transcript or AI panel to search.")
            return
        if self._grep_entry:
            self._grep_entry.set_text(phrase)
        self._apply_grep(phrase)

    def _get_main_text_selection(self) -> str:
        if not self.textview:
            return ""
        return self._get_buffer_selection(self.textview.get_buffer())

    def _get_ai_panel_selection(self) -> str:
        buffer = None
        if self._ai_active_view == AI_VIEW_FILE:
            buffer = self._summary_buffer
        else:
            state = self._ai_outputs.get(self._ai_active_view)
            if state:
                buffer = state.buffer
        return self._get_buffer_selection(buffer)

    def _get_buffer_selection(self, buffer: Gtk.TextBuffer | None) -> str:
        if not buffer:
            return ""
        selection = buffer.get_selection_bounds()
        if not selection:
            return ""
        if len(selection) == 3:
            has_selection, start_iter, end_iter = selection
            if not has_selection:
                return ""
        else:
            start_iter, end_iter = selection
        text = buffer.get_text(start_iter, end_iter, True)
        return text.strip()

    def _go_by(self, delta: int) -> None:
        if not self.pages:
            return
        self._deactivate_continuous_view(reload=False)
        if self._showing_grep_results:
            self._showing_grep_results = False
        new_index = self.current_index + delta
        new_index = max(0, min(len(self.pages) - 1, new_index))
        if new_index != self.current_index:
            self.current_index = new_index
            self._load_current()
        else:
            self._edge_flash()

    def _go_prev(self) -> None:
        self._go_by(-1)

    def _go_next(self) -> None:
        self._go_by(1)

    def _go_prev_ten(self) -> None:
        self._go_by(-10)

    def _go_next_ten(self) -> None:
        self._go_by(10)

    def _go_first(self) -> None:
        if not self.pages:
            return
        self._deactivate_continuous_view(reload=False)
        if self._showing_grep_results:
            self._showing_grep_results = False
        self.current_index = 0
        self._load_current()

    def _go_last(self) -> None:
        if not self.pages:
            return
        self._deactivate_continuous_view(reload=False)
        if self._showing_grep_results:
            self._showing_grep_results = False
        self.current_index = len(self.pages) - 1
        self._load_current()

    def _on_ai_panel_toggled(self, button: Gtk.ToggleButton) -> None:
        self._set_ai_panel_visible(button.get_active())

    def _set_ai_panel_visible(self, visible: bool) -> None:
        if self._ai_panel_revealer:
            self._ai_panel_revealer.set_reveal_child(visible)
        if self._ai_panel_toggle and self._ai_panel_toggle.get_active() != visible:
            self._ai_panel_toggle.set_active(visible)
        self._current_view_state().ai_panel_visible = visible
        if self._ai_panel_icon:
            self._ai_panel_icon.set_from_icon_name(self._ai_panel_icon_name)
        if self._ai_panel_toggle:
            tooltip = (
                "Hide AI panel (Ctrl+Shift+A)"
                if visible
                else "Show AI panel (Ctrl+Shift+A)"
            )
            self._ai_panel_toggle.set_tooltip_text(tooltip)

    def _ensure_ai_panel_visible(self) -> None:
        self._set_ai_panel_visible(True)

    def _set_ai_view(self, view_name: str) -> None:
        target = view_name
        if target not in self._ai_outputs and target != AI_VIEW_FILE:
            target = AI_VIEW_SUMMARIZE
        self._ai_active_view = target
        self._current_view_state().ai_active_view = target
        if target == AI_VIEW_FILE and not self._auto_loading_summary:
            self._auto_load_summary_file()
        if self._ai_view_stack and self._ai_view_stack.get_visible_child_name() != target:
            self._ai_view_stack.set_visible_child_name(target)
        if (
            self._ai_controls_stack
            and self._ai_controls_stack.get_child_by_name(target) is not None
            and self._ai_controls_stack.get_visible_child_name() != target
        ):
            self._ai_controls_stack.set_visible_child_name(target)
        self._sync_ai_view_toggles(target)
        if target in self._ai_outputs:
            state = self._get_ai_output_state(target)
            if state.scroller:
                state.scroller.queue_resize()
        if target == AI_VIEW_FILE and self._summary_scroller:
            self._summary_scroller.queue_resize()
            self._restore_summary_scroll_position(self._summary_loaded_path)

    def _sync_ai_view_toggles(self, target: str) -> None:
        if not self._ai_view_toggles:
            return
        self._ai_view_toggle_guard = True
        try:
            for name, button in self._ai_view_toggles.items():
                button.set_active(name == target)
        finally:
            self._ai_view_toggle_guard = False

    def _on_ai_view_changed(self, stack: Adw.ViewStack, _pspec: GObject.ParamSpec) -> None:
        name = stack.get_visible_child_name() or AI_VIEW_QA
        self._set_ai_view(name)

    def _on_ai_view_toggle(self, button: Gtk.ToggleButton, view_name: str) -> None:
        if self._ai_view_toggle_guard:
            return
        if not button.get_active():
            self._ai_view_toggle_guard = True
            button.set_active(True)
            self._ai_view_toggle_guard = False
            return
        self._set_ai_view(view_name)

    def _on_summarize_page_clicked(self, _button: Gtk.Button) -> None:
        if self._continuous_view or self._showing_grep_results:
            self._transient_toast("Summarize Page only works when a single text page is visible.")
            return
        if not self.pages:
            self._transient_toast("No page loaded to summarize.")
            return
        self._set_ai_view(AI_VIEW_SUMMARIZE)
        page = self.pages[self.current_index]
        content, _, _ = self._read_page_text(page)
        payload = f"Page {page:04d}\n\n{content}"
        self._start_ai_stream(
            label=f"page {page:04d}",
            content=payload,
            prompt_kind="page",
            view_id=self._active_view_id,
        )

    def _on_summarize_range_activate(self, _entry: Gtk.Entry) -> None:
        self._summarize_page_range()

    def _on_summarize_range_button_clicked(self, _button: Gtk.Button) -> None:
        self._summarize_page_range()

    def _summarize_page_range(self) -> None:
        if not self.pages:
            self._transient_toast("No pages available to summarize.")
            return
        if not self._ai_range_entry:
            return
        raw = self._ai_range_entry.get_text().strip()
        page_range = self._parse_page_range(raw)
        if page_range is None:
            self._transient_toast("Enter a page range like 10-25.")
            return
        start_page, end_page = page_range
        targets = [p for p in self.pages if start_page <= p <= end_page]
        if not targets:
            self._transient_toast("No matching pages found in that range.")
            return
        self._set_ai_view(AI_VIEW_SUMMARIZE)
        parts: list[str] = []
        for page in targets:
            content, _, _ = self._read_page_text(page)
            parts.append(f"{page:04d}\n\n{content}\n\n")
        combined = "".join(parts)
        label = f"pages {start_page:04d}-{end_page:04d}"
        self._start_ai_stream(label=label, content=combined, prompt_kind="range", view_id=self._active_view_id)
        self._ai_range_entry.set_text("")

    def _parse_page_range(self, raw: str) -> tuple[int, int] | None:
        if not raw:
            return None
        match = re.fullmatch(r"\s*(\d{1,4})(?:\s*-\s*(\d{1,4}))?\s*", raw)
        if not match:
            return None
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start > end:
            start, end = end, start
        return start, end

    def _on_rag_question_activate(self, _entry: Gtk.Entry) -> None:
        self._submit_rag_question()

    def _on_rag_question_button_clicked(self, _button: Gtk.Button) -> None:
        self._submit_rag_question()

    def _submit_rag_question(self) -> None:
        if not self._rag_question_entry:
            return
        question = self._rag_question_entry.get_text().strip()
        if not question:
            self._transient_toast("Enter a question to run RAG.")
            return
        self._rag_question_entry.set_text("")
        self._start_rag_question(question, self._active_view_id)

    def _on_choose_summary_file_clicked(self, _button: Gtk.Button) -> None:
        self._ensure_ai_panel_visible()
        self._set_ai_view(AI_VIEW_FILE)
        self._open_summary_file_dialog(initial_path=self._summary_file_path or self.input_dir)

    def _open_summary_file_dialog(self, *, initial_path: Path | None = None) -> None:
        if not self.win:
            return
        self._ensure_ai_panel_visible()
        dialog = Gtk.FileDialog()
        dialog.set_title("Select summary file")
        dialog.set_modal(True)
        filters = Gio.ListStore.new(Gtk.FileFilter)
        text_filter = Gtk.FileFilter()
        text_filter.set_name("Text or Markdown")
        text_filter.add_pattern("*.txt")
        text_filter.add_pattern("*.md")
        text_filter.add_mime_type("text/plain")
        text_filter.add_mime_type("text/markdown")
        filters.append(text_filter)
        dialog.set_filters(filters)
        dialog.set_default_filter(text_filter)
        start_path = initial_path
        if start_path is None and self._summary_file_path:
            start_path = self._summary_file_path
        if start_path is None and self.input_dir.exists():
            start_path = self.input_dir
        if start_path:
            folder = start_path if start_path.is_dir() else start_path.parent
            if folder.exists():
                try:
                    dialog.set_initial_folder(Gio.File.new_for_path(str(folder)))
                except (TypeError, AttributeError):
                    pass
        dialog.open(self.win, None, self._on_summary_file_response)
        self._summary_file_dialog = dialog

    def _on_summary_file_response(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        try:
            file = dialog.open_finish(result)
        except GLib.Error as exc:
            if not exc.matches(Gio.io_error_quark(), Gio.IOErrorEnum.CANCELLED):
                self._transient_toast(f"File selection failed: {exc.message}")
        else:
            path_str = file.get_path() if file else None
            if path_str:
                self._load_summary_from_path(Path(path_str))
        if self._summary_file_dialog is dialog:
            self._summary_file_dialog = None
        self._auto_load_summary_file()

    def _auto_load_summary_file(self) -> None:
        if self._auto_loading_summary:
            return
        self._auto_loading_summary = True
        try:
            if not self._summary_file_path:
                self._summary_file_path = load_summary_file_from_config(self.input_dir)
            self._update_summary_buttons()
            if not self._summary_file_path:
                return
            if not self._summary_file_path.exists():
                return
            if (
                self._summary_loaded_path
                and self._summary_loaded_path == self._summary_file_path
                and self._summary_has_text()
            ):
                return
            self._load_summary_from_path(self._summary_file_path, allow_auto=True)
        finally:
            self._auto_loading_summary = False

    def _set_summary_text(self, text: str, *, switch_view: bool = True) -> None:
        self._summary_raw = text or ""
        if self._summary_buffer:
            self._apply_summary_links(text or "")
        self._refresh_summary_search(reset_active=True)
        self._show_summary_view(switch_view=switch_view)

    def _load_summary_from_path(self, path: Path, *, allow_auto: bool = False) -> None:
        if self._auto_loading_summary and not allow_auto:
            # Prevent re-entry if an auto load is already in progress.
            return
        target = path.expanduser()
        resolved = target.resolve(strict=False)
        if not resolved.exists() or not resolved.is_file():
            self._transient_toast(f"File not found: {resolved}")
            return
        try:
            text = resolved.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:  # noqa: BLE001
            self._transient_toast(f"Could not read {resolved.name}: {exc}")
            return
        self._stop_ai_stream_if_running(self._active_view_id)
        self._summary_file_path = resolved
        self._summary_loaded_path = resolved
        save_summary_file_to_config(resolved)
        self._set_summary_text(text, switch_view=not allow_auto)
        self._update_ai_status("", spinning=False, view_id=self._active_view_id)
        self._transient_toast(f"Loaded {resolved.name}")
        self._update_summary_buttons()
        if not allow_auto:
            self._ensure_ai_panel_visible()

    def _start_rag_question(self, question: str, view_id: str | None = None) -> None:
        target_view_id = view_id or self._active_view_id
        state = self._get_view_state(target_view_id)
        self._ai_settings = load_ai_settings()
        settings = self._ai_settings
        rag_api_url, rag_api_key = settings.rag_credentials()
        if not rag_api_key:
            self._transient_toast("Configure the RAG API key in Settings.")
            self._ensure_ai_panel_visible()
            return
        if not settings.rag_llm_model.strip():
            self._transient_toast("Set the RAG answer model in Settings.")
            self._ensure_ai_panel_visible()
            return
        if not settings.voyage_api_key.strip() or not settings.voyage_model.strip():
            self._transient_toast("Set the Voyage API key and model in Settings.")
            self._ensure_ai_panel_visible()
            return
        if not settings.rag_prompt.strip():
            settings.rag_prompt = DEFAULT_RAG_PROMPT
        if not rag_api_url:
            self._transient_toast("Set the RAG API URL in Settings.")
            self._ensure_ai_panel_visible()
            return
        target_view = AI_VIEW_QA
        self._stop_ai_stream_if_running(target_view_id)
        state.ai_cancel_event = threading.Event()
        state.ai_in_flight = True
        state.ai_request_generation += 1
        generation = state.ai_request_generation
        if target_view_id == self._active_view_id:
            self._ai_request_generation = generation
        state.ai_active_view = target_view
        if target_view_id == self._active_view_id:
            self._ai_cancel_event = state.ai_cancel_event
            self._ai_in_flight = True
            self._ensure_ai_panel_visible()
            self._set_ai_view(target_view)
        self._reset_ai_output("", target=target_view, view_id=target_view_id)
        self._update_ai_status("Loading RAG context…", spinning=True, view_id=target_view_id)

        cancel_event = state.ai_cancel_event
        question_text = question.strip()
        label = f"question: {question_text[:48]}{'…' if len(question_text) > 48 else ''}"

        def worker() -> None:
            vectorstore, case_details, error = self._ensure_rag_resources_ready(settings)
            if error or vectorstore is None or case_details is None:
                GLib.idle_add(
                    self._on_ai_stream_error,
                    error or "RAG data unavailable.",
                    generation,
                    target_view,
                    target_view_id,
                )
                return
            try:
                docs = vectorstore.similarity_search(
                    question_text,
                    k=settings.rag_chunk_count,
                )
                context_text = self._format_rag_context(docs)
            except Exception as exc:  # noqa: BLE001
                GLib.idle_add(
                    self._on_ai_stream_error,
                    f"RAG search failed: {exc}",
                    generation,
                    target_view,
                    target_view_id,
                )
                return

            system_prompt = settings.rag_prompt or DEFAULT_RAG_PROMPT
            user_payload = self._compose_rag_payload(case_details, context_text, question_text)
            GLib.idle_add(self._update_ai_status, "Answering question…", True, target_view_id)
            self._stream_chat_worker(
                settings,
                user_payload,
                label,
                cancel_event,
                generation,
                system_prompt,
                target_view,
                target_view_id,
                model_id=settings.rag_llm_model or settings.page_credentials()[1],
                api_url=rag_api_url,
                api_key=rag_api_key,
            )

        state.ai_stream_thread = threading.Thread(target=worker, daemon=True)
        state.ai_stream_thread.start()
        if target_view_id == self._active_view_id:
            self._ai_stream_thread = state.ai_stream_thread

    def _kickoff_rag_background_load(self) -> None:
        settings = self._ai_settings
        if not settings.voyage_api_key.strip() or not settings.voyage_model.strip():
            with self._rag_lock:
                self._rag_vectorstore = None
                self._rag_case_details = None
                self._rag_load_error = "Voyage API key and model are required for RAG."
                self._rag_loading = False
                self._rag_load_thread = None
            return
        self._rag_load_generation += 1
        generation = self._rag_load_generation
        with self._rag_lock:
            self._rag_vectorstore = None
            self._rag_case_details = None
            self._rag_load_error = None
            self._rag_loading = True
        input_dir = self.input_dir
        settings_snapshot = settings

        def worker() -> None:
            store, details, error = self._load_rag_resources(input_dir, settings_snapshot)
            GLib.idle_add(self._on_rag_resources_loaded, generation, store, details, error)

        self._rag_load_thread = threading.Thread(target=worker, daemon=True)
        self._rag_load_thread.start()

    def _on_rag_resources_loaded(
        self,
        generation: int,
        vectorstore: Any | None,
        case_details: str | None,
        error: str | None,
    ) -> bool:
        if generation != self._rag_load_generation:
            return False
        with self._rag_lock:
            if error:
                self._rag_vectorstore = None
                self._rag_case_details = None
                self._rag_load_error = error
            else:
                self._rag_vectorstore = vectorstore
                self._rag_case_details = case_details
                self._rag_load_error = None
            self._rag_loading = False
            self._rag_load_thread = None
        return False

    def _ensure_rag_resources_ready(self, settings: AiSettings) -> tuple[Any | None, str | None, str | None]:
        thread = self._rag_load_thread
        if thread and thread.is_alive():
            thread.join()
        with self._rag_lock:
            if self._rag_vectorstore is not None and self._rag_case_details is not None:
                return self._rag_vectorstore, self._rag_case_details, None
        store, details, error = self._load_rag_resources(self.input_dir, settings)
        if error:
            with self._rag_lock:
                self._rag_load_error = error
            return None, None, error
        with self._rag_lock:
            self._rag_vectorstore = store
            self._rag_case_details = details
            self._rag_load_error = None
            self._rag_loading = False
        return store, details, None

    def _load_rag_resources(
        self,
        input_dir: Path,
        settings: AiSettings,
    ) -> tuple[Any | None, str | None, str | None]:
        embeddings_dir = input_dir / "Embeddings"
        vector_dir = embeddings_dir / "vector_database"
        case_details_path = embeddings_dir / "case_details" / "case_details.txt"
        if not vector_dir.exists() or not vector_dir.is_dir():
            return None, None, f"Vector database not found at {vector_dir}."
        if not case_details_path.exists():
            return None, None, f"Case details file not found at {case_details_path}."
        if not settings.voyage_api_key.strip() or not settings.voyage_model.strip():
            return None, None, "Voyage settings missing."

        try:
            from langchain_chroma import Chroma  # type: ignore
            from langchain_voyageai import VoyageAIEmbeddings  # type: ignore
        except ImportError:
            return None, None, (
                "Install langchain, langchain-chroma, and langchain-voyageai to enable RAG questions."
            )

        try:
            embeddings = VoyageAIEmbeddings(
                voyage_api_key=settings.voyage_api_key,
                model=settings.voyage_model,
            )
            vectorstore = Chroma(persist_directory=str(vector_dir), embedding_function=embeddings)
            case_details = case_details_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:  # noqa: BLE001
            return None, None, f"Failed to load RAG resources: {exc}"

        return vectorstore, case_details, None

    def _format_rag_context(self, docs: list[Any]) -> str:
        chunks: list[str] = []
        for doc in docs:
            try:
                metadata = getattr(doc, "metadata", {}) or {}
                source = ""
                if isinstance(metadata, dict):
                    src_val = metadata.get("source") or metadata.get("page")
                    if src_val:
                        source = str(src_val)
                text = getattr(doc, "page_content", None) or ""
                if source:
                    chunks.append(f"[{source}]\n{text}")
                else:
                    chunks.append(text)
            except Exception:
                continue
        return "\n\n".join(chunks)

    def _compose_rag_payload(self, case_details: str, context: str, question: str) -> str:
        return (
            "Case Details:\n"
            f"{case_details}\n\n"
            "Transcripts:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{question}"
        )

    def _start_ai_stream(self, *, label: str, content: str, prompt_kind: str, view_id: str | None = None) -> None:
        target_view_id = view_id or self._active_view_id
        state = self._get_view_state(target_view_id)
        self._ai_settings = load_ai_settings()
        settings = self._ai_settings
        if not settings.is_configured():
            self._transient_toast("Configure API URL, model, API key, and prompt in Settings.")
            self._ensure_ai_panel_visible()
            return
        if not content.strip():
            self._transient_toast("Nothing to summarize for the requested selection.")
            return
        if prompt_kind == "range":
            prompt = settings.range_prompt or DEFAULT_SUMMARIZATION_PROMPT
            api_url, model_id, api_key = settings.range_credentials()
        else:
            prompt = settings.page_prompt or DEFAULT_SUMMARIZATION_PROMPT
            api_url, model_id, api_key = settings.page_credentials()

        self._stop_ai_stream_if_running(target_view_id)
        state.ai_cancel_event = threading.Event()
        state.ai_in_flight = True
        state.ai_request_generation += 1
        generation = state.ai_request_generation
        if target_view_id == self._active_view_id:
            self._ai_request_generation = generation
        state.ai_active_view = AI_VIEW_SUMMARIZE
        if target_view_id == self._active_view_id:
            self._ai_cancel_event = state.ai_cancel_event
            self._ai_in_flight = True
            self._ensure_ai_panel_visible()
        target_view = AI_VIEW_SUMMARIZE
        if target_view_id == self._active_view_id:
            self._set_ai_view(target_view)
        self._reset_ai_output("", target=target_view, view_id=target_view_id)
        self._update_ai_status(f"Summarizing {label}…", spinning=True, view_id=target_view_id)

        payload_text = content
        worker_settings = settings
        cancel_event = state.ai_cancel_event

        def worker() -> None:
            self._stream_chat_worker(
                worker_settings,
                payload_text,
                label,
                cancel_event,
                generation,
                prompt,
                target_view,
                target_view_id,
                model_id=model_id,
                api_url=api_url,
                api_key=api_key,
            )

        state.ai_stream_thread = threading.Thread(target=worker, daemon=True)
        state.ai_stream_thread.start()
        if target_view_id == self._active_view_id:
            self._ai_stream_thread = state.ai_stream_thread

    def _update_ai_status(self, text: str, spinning: bool, view_id: str | None = None) -> None:
        target_view_id = view_id or self._active_view_id
        state = self._get_view_state(target_view_id)
        state.ai_status_text = text
        state.ai_spinning = spinning
        if target_view_id == self._active_view_id and self._ai_spinner:
            self._ai_spinner.set_spinning(spinning)
            self._ai_spinner.set_visible(spinning)

    def _reset_ai_output(self, text: str | None = None, *, target: str, view_id: str | None = None) -> None:
        target_view_id = view_id or self._active_view_id
        focus_state = self._get_view_state(target_view_id)
        focus_state.ai_output_raw[target] = text or ""
        if target_view_id == self._active_view_id:
            state = self._get_ai_output_state(target)
            self._set_ai_view(target)
            state.raw = text or ""
            self._apply_ai_output_links(state.raw, state)

    def _append_ai_output(self, text: str, generation: int, target: str, view_id: str) -> bool:
        focus_state = self._get_view_state(view_id)
        if generation != focus_state.ai_request_generation:
            return False
        if not text:
            return False
        current_raw = focus_state.ai_output_raw.get(target, "") or ""
        new_raw = current_raw + text
        focus_state.ai_output_raw[target] = new_raw
        if view_id == self._active_view_id:
            state = self._get_ai_output_state(target)
            state.raw = new_raw
            self._apply_ai_output_links(state.raw, state)
        self._update_ai_status("Streaming…", spinning=True, view_id=view_id)
        return False

    def _stop_ai_stream_if_running(self, view_id: str | None = None) -> None:
        target_view_id = view_id or self._active_view_id
        state = self._get_view_state(target_view_id)
        if state.ai_cancel_event:
            state.ai_cancel_event.set()
        if state.ai_stream_thread and state.ai_stream_thread.is_alive():
            try:
                state.ai_stream_thread.join(timeout=0.2)
            except Exception:
                pass
        state.ai_stream_thread = None
        state.ai_cancel_event = None
        state.ai_in_flight = False
        if target_view_id == self._active_view_id:
            self._ai_stream_thread = None
            self._ai_cancel_event = None
            self._ai_in_flight = False

    def _stream_chat_worker(
        self,
        settings: AiSettings,
        content: str,
        label: str,
        cancel_event: threading.Event | None,
        generation: int,
        prompt: str,
        target_view: str,
        target_view_id: str,
        *,
        model_id: str,
        api_url: str,
        api_key: str | None = None,
    ) -> None:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {api_key or settings.api_key}",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Focus/1.0",
        }
        body = {
            "model": model_id,
            "stream": True,
            "messages": [
                {"role": "system", "content": prompt or DEFAULT_SUMMARIZATION_PROMPT},
                {"role": "user", "content": content},
            ],
        }

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                for chunk in self._iter_sse_chunks(resp, cancel_event):
                    if cancel_event and cancel_event.is_set():
                        GLib.idle_add(self._on_ai_stream_cancelled, generation, target_view, target_view_id)
                        return
                    GLib.idle_add(self._append_ai_output, chunk, generation, target_view, target_view_id)
            if cancel_event and cancel_event.is_set():
                GLib.idle_add(self._on_ai_stream_cancelled, generation, target_view, target_view_id)
            else:
                GLib.idle_add(self._on_ai_stream_finished, label, generation, target_view, target_view_id)
        except urllib.error.HTTPError as exc:
            GLib.idle_add(
                self._on_ai_stream_error,
                f"HTTP error {exc.code}: {exc.reason or 'request failed'}",
                generation,
                target_view,
                target_view_id,
            )
        except Exception as exc:  # noqa: BLE001
            GLib.idle_add(self._on_ai_stream_error, str(exc), generation, target_view, target_view_id)

    def _iter_sse_chunks(
        self,
        resp: urllib.response.addinfourl,  # type: ignore[type-arg]
        cancel_event: threading.Event | None,
    ) -> Iterable[str]:
        while True:
            if cancel_event and cancel_event.is_set():
                break
            raw = resp.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].lstrip()
            if data == "[DONE]":
                break
            if not data:
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta_text = self._extract_delta_text(payload)
            if delta_text:
                yield delta_text

    def _extract_delta_text(self, payload: Any) -> str:
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if isinstance(choices, list) and choices:
            first = choices[0] or {}
            delta = first.get("delta") or first.get("message") or first
            if isinstance(delta, dict):
                text = delta.get("content") or delta.get("text")
                if isinstance(text, list):
                    merged = []
                    for item in text:
                        if isinstance(item, dict):
                            merged.append(str(item.get("text", "")))
                        elif isinstance(item, str):
                            merged.append(item)
                    return "".join(merged)
                if isinstance(text, str):
                    return text
        if isinstance(payload, dict):
            fallback = payload.get("data") or payload.get("text")
            if isinstance(fallback, str):
                return fallback
        return ""

    def _on_ai_stream_finished(self, label: str, generation: int, target_view: str, view_id: str) -> bool:
        state = self._get_view_state(view_id)
        if generation != state.ai_request_generation:
            return False
        state.ai_in_flight = False
        state.ai_cancel_event = None
        state.ai_stream_thread = None
        if view_id == self._active_view_id:
            self._ai_in_flight = False
            self._ai_cancel_event = None
            self._ai_stream_thread = None
        self._update_ai_status(f"Finished AI response for {label}.", spinning=False, view_id=view_id)
        return False

    def _on_ai_stream_error(self, message: str, generation: int, target_view: str, view_id: str) -> bool:
        state = self._get_view_state(view_id)
        if generation != state.ai_request_generation:
            return False
        state.ai_in_flight = False
        state.ai_cancel_event = None
        state.ai_stream_thread = None
        if view_id == self._active_view_id:
            self._ai_in_flight = False
            self._ai_cancel_event = None
            self._ai_stream_thread = None
        self._update_ai_status("AI request failed.", spinning=False, view_id=view_id)
        self._transient_toast(message or "AI request failed.")
        return False

    def _on_ai_stream_cancelled(self, generation: int, target_view: str, view_id: str) -> bool:
        state = self._get_view_state(view_id)
        if generation != state.ai_request_generation:
            return False
        state.ai_in_flight = False
        state.ai_cancel_event = None
        state.ai_stream_thread = None
        if view_id == self._active_view_id:
            self._ai_in_flight = False
            self._ai_cancel_event = None
            self._ai_stream_thread = None
        self._update_ai_status("Cancelled.", spinning=False, view_id=view_id)
        return False

    def _edge_flash(self) -> None:
        if not self.win:
            return
        win = self.win
        if self._edge_flash_source_id is not None:
            GLib.source_remove(self._edge_flash_source_id)
            self._edge_flash_source_id = None
        win.remove_css_class("accent")
        win.add_css_class("accent")
        self._edge_flash_source_id = GLib.timeout_add(120, self._edge_flash_reset)

    def _edge_flash_reset(self) -> bool:
        self._edge_flash_source_id = None
        if self.win:
            self.win.remove_css_class("accent")
        return False

    def _transient_toast(self, text: str) -> None:
        if not self.win:
            return
        toast = Adw.Toast.new(text)
        overlay = self._ensure_toast_overlay()
        overlay.add_toast(toast)

    def _ensure_toast_overlay(self) -> Adw.ToastOverlay:
        assert self.win is not None
        content = self.win.get_content()
        if isinstance(content, Adw.ToastOverlay):
            return content
        overlay = Adw.ToastOverlay()
        if content is not None:
            self.win.set_content(None)
            overlay.set_child(content)
        self.win.set_content(overlay)
        return overlay


class TocWindow(Adw.ApplicationWindow):
    def __init__(self, app: Focus):
        super().__init__(application=app)
        self.app = app

        self._buffer_guard = False
        self._buffer_dirty = False
        self._busy = False
        self._toc_load_generation = 0

        self.add_css_class("focus-toc-window")
        self.set_title("Create TOC")
        self.set_default_size(800, 640)

        toolbar_view = Adw.ToolbarView()
        self.set_content(toolbar_view)

        header = Adw.HeaderBar()
        header.add_css_class("flat")
        title = Gtk.Label(label="TOC Creator")
        title.add_css_class("bold-title")
        header.set_title_widget(title)

        toolbar_view.add_top_bar(header)

        content = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin_top=12,
            margin_bottom=12,
            margin_start=12,
            margin_end=12,
        )
        toolbar_view.set_content(content)

        group = Adw.PreferencesGroup()
        group.add_css_class("list-stack")
        group.set_hexpand(True)
        content.append(group)
        self._actions_group = group

        self._create_row = Adw.ActionRow(title="Create TOC")
        self._create_row.add_css_class("list-card")
        self._create_row.set_activatable(True)
        self._create_row.add_prefix(Gtk.Image.new_from_icon_name("view-list-ordered-symbolic"))
        self._create_status = Gtk.Label(label="Idle")
        self._create_spinner = Gtk.Spinner(spinning=False)
        self._create_row.add_suffix(self._create_status)
        self._create_row.add_suffix(self._create_spinner)
        self._create_row.connect("activated", self._on_create_clicked)
        group.add(self._create_row)

        self._dedupe_row = Adw.ActionRow(title="Remove Duplicates")
        self._dedupe_row.add_css_class("list-card")
        self._dedupe_row.set_activatable(True)
        self._dedupe_row.add_prefix(Gtk.Image.new_from_icon_name("list-remove-symbolic"))
        self._dedupe_spinner = Gtk.Spinner(spinning=False)
        self._dedupe_row.add_suffix(self._dedupe_spinner)
        self._dedupe_row.connect("activated", self._on_dedupe_clicked)
        group.add(self._dedupe_row)

        self._save_row = Adw.ActionRow(title="Save TOC")
        self._save_row.add_css_class("list-card")
        self._save_row.set_activatable(True)
        self._save_row.add_prefix(Gtk.Image.new_from_icon_name("document-save-symbolic"))
        self._save_status = Gtk.Label(label="Idle")
        self._save_spinner = Gtk.Spinner(spinning=False)
        self._save_row.add_suffix(self._save_status)
        self._save_row.add_suffix(self._save_spinner)
        self._save_row.connect("activated", self._on_save_clicked)
        group.add(self._save_row)

        paths_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        paths_box.set_hexpand(True)
        content.append(paths_box)

        self._text_dir_label = Gtk.Label(xalign=0)
        self._text_dir_label.set_hexpand(True)
        self._text_dir_label.set_ellipsize(Pango.EllipsizeMode.END)
        text_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        text_row.set_hexpand(True)
        text_row.append(Gtk.Label(label="Text directory:", xalign=0))
        text_row.append(self._text_dir_label)
        paths_box.append(text_row)

        regex_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        regex_row.set_hexpand(True)
        regex_row.append(Gtk.Label(label="Regex directory:", xalign=0))
        self._regex_dir_label = Gtk.Label(xalign=0)
        self._regex_dir_label.set_hexpand(True)
        self._regex_dir_label.set_ellipsize(Pango.EllipsizeMode.END)
        regex_row.append(self._regex_dir_label)
        self._choose_regex_btn = Gtk.Button(label="Choose…")
        self._choose_regex_btn.add_css_class("flat")
        self._choose_regex_btn.connect("clicked", self._on_choose_regex_clicked)
        regex_row.append(self._choose_regex_btn)
        paths_box.append(regex_row)

        editor_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        editor_section.set_vexpand(True)
        editor_section.set_hexpand(True)
        content.append(editor_section)

        header_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        header_row.append(Gtk.Label(label="TOC (editable):", xalign=0))
        editor_section.append(header_row)

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroller.set_hexpand(True)
        scroller.set_vexpand(True)
        scroller.add_css_class("editor-frame")
        editor_section.append(scroller)

        self._text_view = Gtk.TextView()
        self._text_view.set_wrap_mode(Gtk.WrapMode.NONE)
        self._text_view.set_monospace(True)
        self._text_view.set_pixels_above_lines(2)
        self._text_view.set_pixels_below_lines(2)
        self._text_view.add_css_class("editor-textview")
        scroller.set_child(self._text_view)

        self._text_buffer = self._text_view.get_buffer()
        self._text_buffer.connect("changed", self._on_buffer_changed)

        self._status_label = Gtk.Label(label="", xalign=0)
        self._status_label.add_css_class("footer-status")
        self._status_label.set_wrap(True)
        self._status_label.set_hexpand(True)
        editor_section.append(self._status_label)

        self.on_input_dir_changed()
        self.on_regex_dir_changed()
        self._load_existing_toc_async()

    def _toc_path(self) -> Path:
        return self.app.text_dir / "toc.txt"

    def _set_status(self, message: str) -> None:
        self._status_label.set_text(message)

    def _spin(self, kind: str, spinning: bool) -> None:
        if kind == "create":
            self._create_spinner.set_spinning(spinning)
        elif kind == "save":
            self._save_spinner.set_spinning(spinning)
        elif kind == "dedupe":
            self._dedupe_spinner.set_spinning(spinning)

    def _set_row_status(self, kind: str, text: str) -> None:
        if kind == "create":
            self._create_status.set_text(text)
        elif kind == "save":
            self._save_status.set_text(text)

    def _set_busy(self, busy: bool) -> None:
        if self._busy == busy:
            return
        self._busy = busy
        self._create_row.set_sensitive(not busy)
        self._save_row.set_sensitive(not busy)
        self._dedupe_row.set_sensitive(not busy)
        self._choose_regex_btn.set_sensitive(not busy)

    def _replace_buffer_text(self, text: str) -> None:
        self._buffer_guard = True
        self._text_buffer.set_text(text)
        self._buffer_guard = False
        self._buffer_dirty = False
        self._set_row_status("save", "Idle")
        self.app.on_toc_text_updated(text)

    def _get_buffer_text(self) -> str:
        start_iter = self._text_buffer.get_start_iter()
        end_iter = self._text_buffer.get_end_iter()
        return self._text_buffer.get_text(start_iter, end_iter, False)

    def _write_buffer_to_disk(self, mark_clean: bool = True) -> None:
        toc_path = self._toc_path()
        toc_path.parent.mkdir(parents=True, exist_ok=True)
        text = self._get_buffer_text()
        toc_path.write_text(text, encoding="utf-8")
        if mark_clean:
            self._buffer_dirty = False
        self._set_row_status("save", "Saved")
        self.app.on_toc_text_updated(text)

    def _on_create_clicked(self, _row: Adw.ActionRow) -> None:
        if self._busy:
            return
        text_dir = self.app.text_dir
        regex_dir = self.app.regex_dir
        self._set_busy(True)
        self._spin("create", True)
        self._set_row_status("create", "Running…")
        self._set_status("Creating TOC…")

        def worker() -> None:
            try:
                toc_text = generate_toc_text(text_dir, regex_dir)
                toc_path = text_dir / "toc.txt"
                toc_path.parent.mkdir(parents=True, exist_ok=True)
                toc_path.write_text(toc_text, encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                GLib.idle_add(self._on_create_finished, None, exc, text_dir)
                return
            GLib.idle_add(self._on_create_finished, toc_text, None, text_dir)

        threading.Thread(target=worker, daemon=True).start()

    def _on_create_finished(
        self,
        toc_text: str | None,
        error: Exception | None,
        target_dir: Path,
    ) -> bool:
        self._spin("create", False)
        self._set_busy(False)
        if error is not None:
            self._set_row_status("create", "Error")
            self._set_status(f"Create TOC failed: {error}")
            return False
        if target_dir != self.app.text_dir:
            self._set_row_status("create", "Done")
            self._set_status(
                "TOC created for previous directory; current input directory changed."
            )
            return False
        self._set_row_status("create", "Done")
        if toc_text is not None:
            self._replace_buffer_text(toc_text)
            self._set_status("TOC created.")
        else:
            self._set_status("TOC updated.")
        return False

    def _on_save_clicked(self, _row: Adw.ActionRow) -> None:
        if self._busy:
            return
        self._set_row_status("save", "Saving…")
        try:
            self._write_buffer_to_disk(mark_clean=True)
        except OSError as exc:  # noqa: BLE001
            self._set_row_status("save", "Error")
            self._set_status(f"Save failed: {exc}")
            return
        self._set_status(f"Saved {self._toc_path().name}.")

    def _on_choose_regex_clicked(self, _btn: Gtk.Button) -> None:
        self.app.open_regex_dir_dialog(self)

    def _on_buffer_changed(self, _buffer: Gtk.TextBuffer) -> None:
        if self._buffer_guard:
            return
        self._buffer_dirty = True
        self._set_row_status("save", "Modified")

    def _load_existing_toc(self) -> None:
        target_dir = self.app.text_dir
        toc_path = target_dir / "toc.txt"
        self._toc_load_generation += 1
        generation = self._toc_load_generation
        text, error = read_toc_text(toc_path)
        self._apply_loaded_toc(generation, text, error, target_dir)

    def _load_existing_toc_async(self) -> None:
        target_dir = self.app.text_dir
        toc_path = target_dir / "toc.txt"
        self._toc_load_generation += 1
        generation = self._toc_load_generation

        def worker() -> None:
            text, error = read_toc_text(toc_path)
            GLib.idle_add(self._apply_loaded_toc, generation, text, error, target_dir)

        threading.Thread(target=worker, daemon=True).start()

    def _apply_loaded_toc(
        self,
        generation: int,
        text: str,
        error: str | None,
        target_dir: Path,
    ) -> bool:
        if generation != self._toc_load_generation:
            return False
        if target_dir != self.app.text_dir:
            return False
        if error:
            self._set_status(error)
        self._replace_buffer_text(text)
        return False

    def _on_dedupe_clicked(self, _row: Adw.ActionRow) -> None:
        if self._busy:
            return
        try:
            self._set_row_status("save", "Saving…")
            self._write_buffer_to_disk(mark_clean=True)
        except OSError as exc:  # noqa: BLE001
            self._set_row_status("save", "Error")
            self._set_status(f"Failed to save before cleaning TOC: {exc}")
            return
        self._set_status("Removing duplicates…")
        self._set_busy(True)
        self._spin("dedupe", True)

        threading.Thread(target=self._run_dedupe_worker, daemon=True).start()

    def _run_dedupe_worker(self) -> None:
        text_dir = self.app.text_dir
        toc_path = text_dir / "toc.txt"
        try:
            raw = toc_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            GLib.idle_add(self._on_dedupe_finished, None, exc, text_dir)
            return
        try:
            cleaned, removed_exact, removed_minutes = clean_toc_text(raw)
            toc_path.parent.mkdir(parents=True, exist_ok=True)
            toc_path.write_text(cleaned, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            GLib.idle_add(self._on_dedupe_finished, None, exc, text_dir)
            return

        summary = (
            "Duplicates removed"
            f" (exact: {removed_exact}, MINUTES date dupes: {removed_minutes})."
        )
        GLib.idle_add(self._on_dedupe_finished, summary, None, text_dir)

    def _on_dedupe_finished(
        self,
        summary: str | None,
        error: Exception | None,
        target_dir: Path,
    ) -> bool:
        self._spin("dedupe", False)
        self._set_busy(False)
        if error is not None:
            self._set_row_status("save", "Error")
            self._set_status(f"Remove duplicates failed: {error}")
            return False

        if target_dir != self.app.text_dir:
            self._set_row_status("save", "Updated")
            self._set_status(
                "Remove duplicates completed for previous directory; active input directory changed."
            )
            return False

        self._load_existing_toc_async()
        self._set_row_status("save", "Updated")
        self._set_status(summary or "Removed duplicates.")
        return False

    def on_input_dir_changed(self) -> None:
        self._text_dir_label.set_text(str(self.app.text_dir))
        self._load_existing_toc_async()

    def on_regex_dir_changed(self) -> None:
        self._regex_dir_label.set_text(str(self.app.regex_dir))


@dataclass
class SummarizationPromptWidgets:
    api_url_row: Adw.EntryRow
    model_row: Adw.EntryRow
    api_key_row: Adw.EntryRow
    prompt_buffer: Gtk.TextBuffer


@dataclass
class RagPromptWidgets:
    api_url_row: Adw.EntryRow
    model_row: Adw.EntryRow
    api_key_row: Adw.EntryRow
    voyage_model_row: Adw.EntryRow
    voyage_key_row: Adw.EntryRow
    rag_chunk_row: Adw.SpinRow
    prompt_buffer: Gtk.TextBuffer


class AiSettingsWindow(Adw.ApplicationWindow):
    def __init__(self, app: Focus):
        super().__init__(application=app)
        self.app = app

        self._status_label: Gtk.Label | None = None
        self._record_font_size_row: Adw.SpinRow | None = None
        self._ai_font_size_row: Adw.SpinRow | None = None
        self._highlight_phrases_buffer: Gtk.TextBuffer | None = None
        self._prompt_editors: dict[str, SummarizationPromptWidgets | RagPromptWidgets] = {}
        self._prompt_row_keys: dict[Gtk.ListBoxRow, str] = {}
        self._prompt_list: Gtk.ListBox | None = None
        self._prompt_stack: Gtk.Stack | None = None

        self.set_title("Settings")
        self.set_default_size(900, 720)
        self.set_resizable(True)
        self._build_ui()
        self._load_settings()

    def _build_ui(self) -> None:
        view = Adw.ToolbarView()
        header = Adw.HeaderBar()
        header.add_css_class("flat")
        header.set_title_widget(Gtk.Label(label="Settings", xalign=0))
        view.add_top_bar(header)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_top(18)
        box.set_margin_bottom(12)
        box.set_margin_start(18)
        box.set_margin_end(18)

        display_group = Adw.PreferencesGroup(title="Display")
        display_group.add_css_class("list-stack")
        display_group.set_hexpand(True)
        box.append(display_group)

        ai_font_adjustment = Gtk.Adjustment(
            value=self.app.get_font_preferences()[1],
            lower=8,
            upper=48,
            step_increment=1,
            page_increment=2,
        )
        self._ai_font_size_row = Adw.SpinRow(
            title="AI Panel Font Size (pt)",
            adjustment=ai_font_adjustment,
        )
        self._ai_font_size_row.set_digits(0)
        display_group.add(self._ai_font_size_row)

        base_font_adjustment = Gtk.Adjustment(
            value=self.app.get_font_preferences()[0],
            lower=8,
            upper=48,
            step_increment=1,
            page_increment=2,
        )
        self._record_font_size_row = Adw.SpinRow(
            title="Record Font Size (pt)",
            adjustment=base_font_adjustment,
        )
        self._record_font_size_row.set_digits(0)
        display_group.add(self._record_font_size_row)

        highlight_group = Adw.PreferencesGroup(title="Highlights")
        highlight_group.add_css_class("list-stack")
        highlight_group.set_hexpand(True)
        box.append(highlight_group)

        highlight_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        highlight_box.set_margin_top(6)
        highlight_box.set_margin_bottom(6)
        highlight_box.set_margin_start(12)
        highlight_box.set_margin_end(12)
        highlight_label = Gtk.Label(
            label="Highlight phrases (case-sensitive, one per line)",
            xalign=0,
        )
        highlight_label.add_css_class("dim-label")
        highlight_box.append(highlight_label)

        highlight_scroller = Gtk.ScrolledWindow()
        highlight_scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        highlight_scroller.set_hexpand(True)
        highlight_scroller.set_vexpand(False)
        highlight_scroller.set_min_content_height(110)
        highlight_buffer = Gtk.TextBuffer()
        highlight_view = Gtk.TextView.new_with_buffer(highlight_buffer)
        highlight_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        highlight_view.set_top_margin(8)
        highlight_view.set_bottom_margin(8)
        highlight_view.set_left_margin(8)
        highlight_view.set_right_margin(8)
        highlight_scroller.set_child(highlight_view)
        highlight_box.append(highlight_scroller)
        highlight_group.add(highlight_box)
        self._highlight_phrases_buffer = highlight_buffer

        split = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        split.set_hexpand(True)
        split.set_vexpand(True)
        split.set_shrink_start_child(False)
        split.set_shrink_end_child(False)
        split.set_resize_start_child(False)
        split.set_resize_end_child(True)

        prompt_list = Gtk.ListBox()
        prompt_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        prompt_list.add_css_class("navigation-sidebar")
        prompt_list.connect("row-selected", self._on_prompt_row_selected)
        self._prompt_list = prompt_list

        prompt_list_scroller = Gtk.ScrolledWindow()
        prompt_list_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        prompt_list_scroller.set_min_content_width(240)
        prompt_list_scroller.set_child(prompt_list)

        prompt_stack = Gtk.Stack()
        prompt_stack.set_hexpand(True)
        prompt_stack.set_vexpand(True)
        prompt_stack.set_transition_type(Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)
        self._prompt_stack = prompt_stack

        prompt_definitions = [
            ("page", "Single Page Summarization", self._build_summarization_prompt_page),
            ("range", "Page Range Summarization", self._build_summarization_prompt_page),
            ("rag", "RAG Answer Prompt", self._build_rag_prompt_page),
        ]
        first_row: Gtk.ListBoxRow | None = None
        for key, title, builder in prompt_definitions:
            row = Gtk.ListBoxRow()
            row_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            row_box.set_margin_top(8)
            row_box.set_margin_bottom(8)
            row_box.set_margin_start(12)
            row_box.set_margin_end(12)
            label = Gtk.Label(label=title, xalign=0)
            row_box.append(label)
            row.set_child(row_box)
            prompt_list.append(row)
            self._prompt_row_keys[row] = key
            if first_row is None:
                first_row = row

            page = builder(key, title)
            prompt_stack.add_named(page, key)

        if first_row is not None:
            prompt_list.select_row(first_row)
            prompt_stack.set_visible_child_name(self._prompt_row_keys[first_row])

        split.set_start_child(prompt_list_scroller)
        split.set_end_child(prompt_stack)
        box.append(split)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)
        scrolled.set_child(box)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        content.append(scrolled)

        buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        buttons.set_margin_top(6)
        buttons.set_margin_bottom(12)
        buttons.set_margin_start(12)
        buttons.set_margin_end(12)
        buttons.set_halign(Gtk.Align.END)
        save_btn = Gtk.Button(label="Save Settings")
        save_btn.add_css_class("suggested-action")
        save_btn.add_css_class("flat")
        save_btn.connect("clicked", self._on_save_clicked)
        buttons.append(save_btn)
        content.append(buttons)

        self._status_label = Gtk.Label(label="", xalign=0)
        self._status_label.set_wrap(True)
        self._status_label.add_css_class("dim-label")
        content.append(self._status_label)

        view.set_content(content)
        self.set_content(view)

    def _build_password_row(self, title: str) -> Adw.EntryRow:
        password_row_cls = getattr(Adw, "PasswordEntryRow", None)
        if password_row_cls:
            row = password_row_cls(title=title)
            if hasattr(row, "set_show_peek_icon"):
                row.set_show_peek_icon(True)
        else:
            row = Adw.EntryRow(title=title)
            if hasattr(row, "set_input_purpose"):
                row.set_input_purpose(Gtk.InputPurpose.PASSWORD)
            if hasattr(row, "set_visibility"):
                try:
                    row.set_visibility(False)
                except Exception:
                    pass
        if hasattr(row, "set_hexpand"):
            row.set_hexpand(True)
        return row

    def _build_prompt_editor(self, text: str) -> tuple[Gtk.ScrolledWindow, Gtk.TextBuffer]:
        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroller.set_hexpand(True)
        scroller.set_vexpand(True)
        scroller.set_has_frame(False)

        buffer = Gtk.TextBuffer()
        buffer.set_text(text)
        prompt_view = Gtk.TextView.new_with_buffer(buffer)
        prompt_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        prompt_view.set_monospace(True)
        prompt_view.set_vexpand(True)
        prompt_view.set_hexpand(True)
        prompt_view.set_top_margin(12)
        prompt_view.set_bottom_margin(12)
        prompt_view.set_left_margin(12)
        prompt_view.set_right_margin(12)
        scroller.set_child(prompt_view)
        return scroller, buffer

    def _build_summarization_prompt_page(self, key: str, title: str) -> Gtk.Widget:
        page_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        page_box.set_margin_top(12)
        page_box.set_margin_bottom(12)
        page_box.set_margin_start(12)
        page_box.set_margin_end(12)
        page_box.set_vexpand(True)

        title_label = Gtk.Label(label=title, xalign=0)
        title_label.add_css_class("title-3")
        page_box.append(title_label)

        credentials_group = Adw.PreferencesGroup(title="Credentials")
        credentials_group.add_css_class("list-stack")
        credentials_group.set_hexpand(True)
        page_box.append(credentials_group)

        api_url_row = Adw.EntryRow(title="API URL")
        api_url_row.set_hexpand(True)
        credentials_group.add(api_url_row)

        model_row = Adw.EntryRow(title="Model ID")
        model_row.set_hexpand(True)
        credentials_group.add(model_row)

        api_key_row = self._build_password_row("API Key")
        credentials_group.add(api_key_row)

        prompt_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        prompt_section.set_hexpand(True)
        prompt_section.set_vexpand(True)
        prompt_label = Gtk.Label(label="Prompt", xalign=0)
        prompt_label.add_css_class("dim-label")
        prompt_section.append(prompt_label)
        prompt_scroller, buffer = self._build_prompt_editor(DEFAULT_SUMMARIZATION_PROMPT)
        prompt_section.append(prompt_scroller)
        page_box.append(prompt_section)

        page = Gtk.ScrolledWindow()
        page.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        page.set_hexpand(True)
        page.set_vexpand(True)
        page.set_child(page_box)

        self._prompt_editors[key] = SummarizationPromptWidgets(
            api_url_row=api_url_row,
            model_row=model_row,
            api_key_row=api_key_row,
            prompt_buffer=buffer,
        )
        return page

    def _build_rag_prompt_page(self, key: str, title: str) -> Gtk.Widget:
        page_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        page_box.set_margin_top(12)
        page_box.set_margin_bottom(12)
        page_box.set_margin_start(12)
        page_box.set_margin_end(12)
        page_box.set_vexpand(True)

        title_label = Gtk.Label(label=title, xalign=0)
        title_label.add_css_class("title-3")
        page_box.append(title_label)

        rag_group = Adw.PreferencesGroup(title="RAG Credentials")
        rag_group.add_css_class("list-stack")
        rag_group.set_hexpand(True)
        page_box.append(rag_group)

        rag_api_url_row = Adw.EntryRow(title="RAG API URL")
        rag_api_url_row.set_hexpand(True)
        rag_group.add(rag_api_url_row)

        rag_api_key_row = self._build_password_row("RAG API Key")
        rag_group.add(rag_api_key_row)

        rag_model_row = Adw.EntryRow(title="RAG Answer Model")
        rag_model_row.set_hexpand(True)
        rag_group.add(rag_model_row)

        rag_context_group = Adw.PreferencesGroup(title="RAG Context")
        rag_context_group.add_css_class("list-stack")
        rag_context_group.set_hexpand(True)
        page_box.append(rag_context_group)

        rag_chunk_adjustment = Gtk.Adjustment(
            value=DEFAULT_RAG_CHUNK_COUNT,
            lower=1,
            upper=50,
            step_increment=1,
            page_increment=2,
        )
        rag_chunk_row = Adw.SpinRow(
            title="Context Chunks",
            adjustment=rag_chunk_adjustment,
        )
        rag_chunk_row.set_digits(0)
        rag_context_group.add(rag_chunk_row)

        voyage_group = Adw.PreferencesGroup(title="Voyage Embeddings")
        voyage_group.add_css_class("list-stack")
        voyage_group.set_hexpand(True)
        page_box.append(voyage_group)

        voyage_model_row = Adw.EntryRow(title="Voyage Embedding Model")
        voyage_model_row.set_hexpand(True)
        voyage_group.add(voyage_model_row)

        voyage_key_row = self._build_password_row("Voyage API Key")
        voyage_group.add(voyage_key_row)

        prompt_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        prompt_section.set_hexpand(True)
        prompt_section.set_vexpand(True)
        prompt_label = Gtk.Label(label="Prompt", xalign=0)
        prompt_label.add_css_class("dim-label")
        prompt_section.append(prompt_label)
        prompt_scroller, buffer = self._build_prompt_editor(DEFAULT_RAG_PROMPT)
        prompt_section.append(prompt_scroller)
        page_box.append(prompt_section)

        page = Gtk.ScrolledWindow()
        page.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        page.set_hexpand(True)
        page.set_vexpand(True)
        page.set_child(page_box)

        self._prompt_editors[key] = RagPromptWidgets(
            api_url_row=rag_api_url_row,
            model_row=rag_model_row,
            api_key_row=rag_api_key_row,
            voyage_model_row=voyage_model_row,
            voyage_key_row=voyage_key_row,
            rag_chunk_row=rag_chunk_row,
            prompt_buffer=buffer,
        )
        return page

    def _on_prompt_row_selected(self, _listbox: Gtk.ListBox, row: Gtk.ListBoxRow | None) -> None:
        if not row or not self._prompt_stack:
            return
        key = self._prompt_row_keys.get(row)
        if key:
            self._prompt_stack.set_visible_child_name(key)

    def _prompt_text(self, buffer: Gtk.TextBuffer) -> str:
        start, end = buffer.get_bounds()
        return buffer.get_text(start, end, True)

    def _load_settings(self) -> None:
        settings = load_ai_settings()
        page_widgets = self._prompt_editors.get("page")
        range_widgets = self._prompt_editors.get("range")
        rag_widgets = self._prompt_editors.get("rag")

        if isinstance(page_widgets, SummarizationPromptWidgets):
            page_widgets.api_url_row.set_text(settings.page_api_url or settings.api_url)
            page_widgets.model_row.set_text(settings.page_model_id or settings.model_id)
            page_widgets.api_key_row.set_text(settings.page_api_key or settings.api_key)
            page_widgets.prompt_buffer.set_text(settings.page_prompt or DEFAULT_SUMMARIZATION_PROMPT)

        if isinstance(range_widgets, SummarizationPromptWidgets):
            range_widgets.api_url_row.set_text(settings.range_api_url or settings.api_url)
            range_widgets.model_row.set_text(settings.range_model_id or settings.model_id)
            range_widgets.api_key_row.set_text(settings.range_api_key or settings.api_key)
            range_widgets.prompt_buffer.set_text(settings.range_prompt or DEFAULT_SUMMARIZATION_PROMPT)

        if isinstance(rag_widgets, RagPromptWidgets):
            rag_widgets.api_url_row.set_text(
                settings.rag_api_url or settings.page_api_url or settings.api_url
            )
            rag_widgets.api_key_row.set_text(
                settings.rag_api_key or settings.page_api_key or settings.api_key
            )
            rag_widgets.model_row.set_text(settings.rag_llm_model)
            rag_widgets.voyage_model_row.set_text(settings.voyage_model)
            rag_widgets.voyage_key_row.set_text(settings.voyage_api_key)
            rag_widgets.rag_chunk_row.set_value(float(settings.rag_chunk_count))
            rag_widgets.prompt_buffer.set_text(settings.rag_prompt or DEFAULT_RAG_PROMPT)

        if self._ai_font_size_row:
            _, ai_font = self.app.get_font_preferences()
            self._ai_font_size_row.set_value(float(ai_font))
        if self._record_font_size_row:
            base_font, _ = self.app.get_font_preferences()
            self._record_font_size_row.set_value(float(base_font))
        if self._highlight_phrases_buffer is not None:
            self._highlight_phrases_buffer.set_text(
                _format_highlight_phrases(settings.highlight_phrases)
            )
        self._set_status("Loaded saved values.")

    def _set_status(self, text: str) -> None:
        if self._status_label:
            self._status_label.set_text(text)

    def _on_save_clicked(self, _btn: Gtk.Button) -> None:
        page_widgets = self._prompt_editors.get("page")
        range_widgets = self._prompt_editors.get("range")
        rag_widgets = self._prompt_editors.get("rag")
        if not isinstance(page_widgets, SummarizationPromptWidgets):
            return
        if not isinstance(range_widgets, SummarizationPromptWidgets):
            return
        if not isinstance(rag_widgets, RagPromptWidgets):
            return

        page_api_url = page_widgets.api_url_row.get_text().strip()
        page_model_id = page_widgets.model_row.get_text().strip()
        page_api_key = page_widgets.api_key_row.get_text().strip()
        range_api_url = range_widgets.api_url_row.get_text().strip()
        range_model_id = range_widgets.model_row.get_text().strip()
        range_api_key = range_widgets.api_key_row.get_text().strip()
        rag_api_url = rag_widgets.api_url_row.get_text().strip()
        rag_api_key = rag_widgets.api_key_row.get_text().strip()
        rag_model = rag_widgets.model_row.get_text().strip()
        voyage_model = rag_widgets.voyage_model_row.get_text().strip()
        voyage_key = rag_widgets.voyage_key_row.get_text().strip()
        rag_chunk_count = _coerce_rag_chunk_count(
            int(round(rag_widgets.rag_chunk_row.get_value())),
            DEFAULT_RAG_CHUNK_COUNT,
        )

        page_prompt = self._prompt_text(page_widgets.prompt_buffer).strip()
        range_prompt = self._prompt_text(range_widgets.prompt_buffer).strip()
        rag_prompt = self._prompt_text(rag_widgets.prompt_buffer).strip()
        highlight_phrases = (
            _normalize_highlight_phrases(self._prompt_text(self._highlight_phrases_buffer))
            if self._highlight_phrases_buffer is not None
            else []
        )

        record_font_size = (
            int(round(self._record_font_size_row.get_value()))
            if self._record_font_size_row
            else self.app.get_font_preferences()[0]
        )
        ai_font_size = (
            int(round(self._ai_font_size_row.get_value()))
            if self._ai_font_size_row
            else self.app.get_font_preferences()[1]
        )
        settings = AiSettings(
            api_url=page_api_url,
            model_id=page_model_id,
            api_key=page_api_key,
            page_api_url=page_api_url,
            page_model_id=page_model_id,
            page_api_key=page_api_key,
            range_api_url=range_api_url,
            range_model_id=range_model_id,
            range_api_key=range_api_key,
            page_prompt=page_prompt or DEFAULT_SUMMARIZATION_PROMPT,
            range_prompt=range_prompt or DEFAULT_SUMMARIZATION_PROMPT,
            voyage_api_key=voyage_key,
            voyage_model=voyage_model or "voyage-law-2",
            rag_llm_model=rag_model,
            rag_prompt=rag_prompt or DEFAULT_RAG_PROMPT,
            rag_api_url=rag_api_url or page_api_url,
            rag_api_key=rag_api_key or page_api_key,
            rag_chunk_count=rag_chunk_count,
            highlight_phrases=highlight_phrases,
        )
        save_ai_settings(settings)
        self.app.update_font_sizes(font_size_pt=record_font_size, ai_font_size_pt=ai_font_size)
        self.app.on_ai_settings_saved(settings)
        if settings.is_configured() and settings.is_rag_ready():
            self._set_status("Saved. Summaries and RAG questions are enabled.")
        elif settings.is_configured():
            self._set_status("Saved. Add RAG API URL, Voyage, and RAG fields to enable questions.")
        else:
            self._set_status("Saved. Add required fields to enable summaries.")



def _prepare_cli_input_dir(raw_path: str) -> Path:
    target = Path(raw_path).expanduser().resolve(strict=False)
    if not target.exists() or not target.is_dir():
        _cli_error(f"Input directory not found: {target}")
    normalized = _normalize_input_dir(target)
    if not normalized.exists() or not normalized.is_dir():
        _cli_error(f"Input directory not found: {normalized}")
    text_dir = _text_dir_from_root(normalized)
    if not text_dir.exists() or not text_dir.is_dir():
        _cli_error(f"'text_record' directory not found inside: {normalized}")
    return normalized


def _cli_error(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)



def main() -> None:
    args = sys.argv[1:]
    input_override: Path | None = None
    if args:
        first = args[0]
        if first in {"-h", "--help"}:
            print("Usage: python focus.py [DIRECTORY]")
            print("Provide DIRECTORY to override the configured record root.")
            return
        if len(args) > 1:
            print("Only one directory argument is supported.", file=sys.stderr)
            raise SystemExit(2)
        input_override = _prepare_cli_input_dir(first)
    app = Focus(input_override=input_override)
    raise SystemExit(app.run(None))


if __name__ == "__main__":
    main()
