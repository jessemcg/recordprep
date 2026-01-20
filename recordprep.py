#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import threading
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk  # type: ignore

import fitz
import pdftotext
from pypdf import PdfReader, PdfWriter

APPLICATION_ID = "com.mcglaw.RecordPrep"
APPLICATION_NAME = "Record Prep"

GLib.set_application_name(APPLICATION_NAME)

CONFIG_FILE = Path(__file__).with_name("config.json")
CONFIG_KEY_CLASSIFIER_API_URL = "classifier_api_url"
CONFIG_KEY_CLASSIFIER_MODEL_ID = "classifier_model_id"
CONFIG_KEY_CLASSIFIER_API_KEY = "classifier_api_key"
CONFIG_KEY_CLASSIFIER_PROMPT = "classifier_prompt"
CONFIG_KEY_CLASSIFY_DATES_API_URL = "classify_dates_api_url"
CONFIG_KEY_CLASSIFY_DATES_MODEL_ID = "classify_dates_model_id"
CONFIG_KEY_CLASSIFY_DATES_API_KEY = "classify_dates_api_key"
CONFIG_KEY_CLASSIFY_DATES_PROMPT = "classify_dates_prompt"
CONFIG_KEY_CLASSIFY_REPORTS_API_URL = "classify_report_names_api_url"
CONFIG_KEY_CLASSIFY_REPORTS_MODEL_ID = "classify_report_names_model_id"
CONFIG_KEY_CLASSIFY_REPORTS_API_KEY = "classify_report_names_api_key"
CONFIG_KEY_CLASSIFY_REPORTS_PROMPT = "classify_report_names_prompt"
CONFIG_KEY_CASE_NAME_API_URL = "case_name_api_url"
CONFIG_KEY_CASE_NAME_MODEL_ID = "case_name_model_id"
CONFIG_KEY_CASE_NAME_API_KEY = "case_name_api_key"
CONFIG_KEY_CASE_NAME_PROMPT = "case_name_prompt"
CONFIG_KEY_CASE_NAME = "case_name"
CONFIG_KEY_CASE_ROOT_DIR = "case_root_dir"
CONFIG_KEY_CLASSIFY_FORMS_API_URL = "classify_form_names_api_url"
CONFIG_KEY_CLASSIFY_FORMS_MODEL_ID = "classify_form_names_model_id"
CONFIG_KEY_CLASSIFY_FORMS_API_KEY = "classify_form_names_api_key"
CONFIG_KEY_CLASSIFY_FORMS_PROMPT = "classify_form_names_prompt"
CONFIG_KEY_OPTIMIZE_API_URL = "optimize_api_url"
CONFIG_KEY_OPTIMIZE_MODEL_ID = "optimize_model_id"
CONFIG_KEY_OPTIMIZE_API_KEY = "optimize_api_key"
CONFIG_KEY_OPTIMIZE_ATTORNEYS_PROMPT = "optimize_attorneys_prompt"
CONFIG_KEY_OPTIMIZE_HEARINGS_PROMPT = "optimize_hearings_prompt"
CONFIG_KEY_OPTIMIZE_REPORTS_PROMPT = "optimize_reports_prompt"
CONFIG_KEY_SUMMARIZE_API_URL = "summarize_api_url"
CONFIG_KEY_SUMMARIZE_MODEL_ID = "summarize_model_id"
CONFIG_KEY_SUMMARIZE_API_KEY = "summarize_api_key"
CONFIG_KEY_SUMMARIZE_HEARINGS_PROMPT = "summarize_hearings_prompt"
CONFIG_KEY_SUMMARIZE_REPORTS_PROMPT = "summarize_reports_prompt"
CONFIG_KEY_SUMMARIZE_CHUNK_SIZE = "summarize_chunk_size"
CONFIG_KEY_SELECTED_PDFS = "selected_pdfs"
DEFAULT_CLASSIFIER_PROMPT = (
    "You are labeling a single page of an OCR'd legal transcript. "
    "Return JSON with keys: page_type, form_name, date. "
    "page_type must be one of: hearing, report, form, cover, index, notice, minute_order. "
    "date should be a long-form U.S. date if present. "
    "form_name should be the form name if page_type is form. "
    "If unknown, use an empty string."
)
DEFAULT_CLASSIFY_DATES_PROMPT = (
    "You are extracting the date from a hearing or minute order page of an OCR'd legal transcript. "
    "Return JSON with keys: date. "
    "date should be a long-form U.S. date if present. "
    "If unknown, use an empty string."
)
DEFAULT_CLASSIFY_REPORT_NAMES_PROMPT = (
    "You are extracting the report name from the first page of a report in an OCR'd legal transcript. "
    "Return JSON with keys: report_name. "
    "report_name should be the formal report title if present. "
    "If unknown, use an empty string."
)
DEFAULT_CLASSIFY_FORM_NAMES_PROMPT = (
    "You are extracting the form name from a form page in an OCR'd legal transcript. "
    "Return JSON with keys: form_name. "
    "form_name should be the formal form title if present. "
    "If unknown, use an empty string."
)
DEFAULT_CASE_NAME_PROMPT = (
    "You are inferring the case name from the first three pages of an OCR'd legal transcript. "
    "Return only the case name as plain text. "
    "The case name should replace spaces with underscores, like In_re_Mark_T or "
    "Social_Services_v_Breanna_F. "
    "If unknown, use an empty string."
)
DEFAULT_OPTIMIZE_ATTORNEYS_PROMPT = (
    "You are reviewing an excerpt from an OCR'd hearing transcript. "
    "Identify the attorneys who appear and who they represent. "
    "Respond with 1-3 narrative sentences. "
    "Do not use lists or markdown. "
    "If representation is unclear, say so briefly."
)
DEFAULT_OPTIMIZE_HEARINGS_PROMPT = (
    "You are reformatting a hearing transcript chunk for retrieval. "
    "Use the exact words from the transcript. "
    "Organize the output into paragraphs of about five sentences each. "
    "Each paragraph must be on a single line and separated by a blank line. "
    "Each paragraph must begin with 'Hearing date: <date>.' followed by a space, "
    "then label each speaker before their dialogue using sentence case "
    "(for example, 'Ms. Smith speaking: ...'). "
    "The transcript may be all caps; convert dialogue to normal sentence case "
    "while preserving words exactly and ensuring pronoun 'I' is capitalized. "
    "Do not add commentary or change wording."
)
DEFAULT_OPTIMIZE_REPORTS_PROMPT = (
    "Reproduce the meaningful portions of the report text verbatim. "
    "Organize the output into paragraphs of about five sentences each. "
    "Each paragraph must be on a single line and separated by a blank line. "
    "Each paragraph must begin with 'Reporting:' followed by a space and the verbatim text. "
    "Do not add commentary or change wording."
)
DEFAULT_SUMMARIZE_HEARINGS_PROMPT = (
    "Summarize the following court hearing in one very concise paragraph using plain "
    "and simple English. Include short direct quotes (3-6 words) from the hearing to "
    "highlight legally significant statements. Each quote must be in quotation marks "
    "and must be verbatim. Do not use ellipses. Do not add commentary or markdown. "
    "Do not begin with prefatory language. Do not include the hearing date in the summary. "
    "Here is the hearing:"
)
DEFAULT_SUMMARIZE_REPORTS_PROMPT = (
    "Summarize the following reports in one very concise paragraph using plain "
    "and simple English. Include short direct quotes (5-10 words) from the reports to "
    "highlight legally significant statements. Each quote must be in quotation marks "
    "and must be verbatim. Do not use ellipses. Do not add commentary or markdown. "
    "Do not begin with prefatory language. Here are the reports:"
)
DEFAULT_SUMMARIZE_CHUNK_SIZE = 15


def _unique_in_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _extract_prompt_keys(prompt: str) -> list[str]:
    if not prompt:
        return []
    lower_prompt = prompt.lower()
    markers = ("names of the keys are", "keys are", "keys:")
    for marker in markers:
        index = lower_prompt.find(marker)
        if index == -1:
            continue
        segment = prompt[index:]
        segment = segment.splitlines()[0]
        if "." in segment:
            segment = segment.split(".", 1)[0]
        tokens = re.findall(r"['\"]([A-Za-z0-9_]+)['\"]", segment)
        if tokens:
            return _unique_in_order(tokens)
    tokens = re.findall(r"['\"]([A-Za-z0-9_]+)['\"]", prompt)
    likely = [token for token in tokens if token.upper() == token or "_" in token]
    if likely:
        return _unique_in_order(likely)
    return _unique_in_order(tokens)


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", key.lower())


def _extract_page_number(filename: str) -> int | None:
    match = re.search(r"(\d+)", Path(filename).stem)
    if match:
        return int(match.group(1))
    return None


def _load_classify_date_targets(classify_path: Path) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    if not classify_path.exists():
        return targets
    entries: list[tuple[str, str, int]] = []
    with classify_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            file_name = str(payload.get("file_name", "") or "").strip()
            page_type = str(payload.get("page_type", "") or "").strip().lower()
            if not file_name or not page_type:
                continue
            page_number = _extract_page_number(file_name)
            if page_number is None:
                continue
            entries.append((file_name, page_type, page_number))
    prev_type: str | None = None
    prev_number: int | None = None
    for file_name, page_type, page_number in entries:
        if page_type in {"hearing", "minute_order"}:
            if page_type != prev_type or prev_number is None or page_number != prev_number + 1:
                targets.append((file_name, page_type))
            prev_type = page_type
            prev_number = page_number
        else:
            prev_type = None
            prev_number = None
    return targets


def _load_classify_report_targets(classify_path: Path) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    if not classify_path.exists():
        return targets
    entries: list[tuple[str, str, int]] = []
    with classify_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            file_name = str(payload.get("file_name", "") or "").strip()
            page_type = str(payload.get("page_type", "") or "").strip().lower()
            if not file_name or not page_type:
                continue
            page_number = _extract_page_number(file_name)
            if page_number is None:
                continue
            entries.append((file_name, page_type, page_number))
    prev_type: str | None = None
    prev_number: int | None = None
    for file_name, page_type, page_number in entries:
        if page_type == "report":
            if prev_type != "report" or prev_number is None or page_number != prev_number + 1:
                targets.append((file_name, page_type))
            prev_type = page_type
            prev_number = page_number
        else:
            prev_type = None
            prev_number = None
    return targets


def _load_classify_form_targets(classify_path: Path) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    if not classify_path.exists():
        return targets
    with classify_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            file_name = str(payload.get("file_name", "") or "").strip()
            page_type = str(payload.get("page_type", "") or "").strip().lower()
            if not file_name or page_type != "form":
                continue
            targets.append((file_name, page_type))
    return targets


def _load_jsonl_entries(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                entries.append(payload)
    return entries


def _load_json_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    return []


def _extract_entry_value(entry: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key in entry:
            value = entry.get(key)
            return str(value).strip() if value is not None else ""
    normalized = {_normalize_key(key): key for key in entry}
    for key in keys:
        normalized_key = _normalize_key(key)
        source_key = normalized.get(normalized_key)
        if source_key is not None:
            value = entry.get(source_key)
            return str(value).strip() if value is not None else ""
    return ""


def _page_label_from_filename(filename: str) -> str:
    return Path(filename).stem if filename else ""


def _page_number_from_label(label: str) -> int | None:
    if not label:
        return None
    return _extract_page_number(label)


def _format_toc_line(label: str, page: str) -> str:
    if label and page:
        return f"\t{label} {page}"
    if label:
        return f"\t{label}"
    if page:
        return f"\t{page}"
    return "\t"


def _sanitize_case_name_tokens(tokens: list[str]) -> str:
    cleaned: list[str] = []
    for token in tokens:
        lowered = token.lower()
        if lowered in {"v", "vs"}:
            cleaned.append("v")
        elif lowered == "re":
            cleaned.append("re")
        elif lowered == "in":
            cleaned.append("In")
        else:
            cleaned.append(token)
    return "_".join(cleaned)


def _sanitize_case_name_value(value: str) -> str:
    trimmed = value.strip()
    if not trimmed:
        return ""
    cleaned = re.sub(r"\s+", "_", trimmed)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def _strip_nonstandard_characters(text: str) -> str:
    cleaned_chars: list[str] = []
    for ch in text:
        if ch in {"\n", "\t"}:
            cleaned_chars.append(ch)
        elif unicodedata.category(ch) != "Cc":
            cleaned_chars.append(ch)
    return "".join(cleaned_chars)


def _split_tagged_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_label: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        match = re.match(r"\s*<<<\s*(.*?)\s*>>>\s*$", line)
        if match:
            if current_label is not None:
                sections.append((current_label, "\n".join(current_lines).strip()))
            current_label = match.group(1).strip() or "Unknown"
            current_lines = []
        else:
            current_lines.append(line)
    if current_label is not None:
        sections.append((current_label, "\n".join(current_lines).strip()))
    return sections


def _split_into_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def _chunk_sentences(sentences: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).rstrip() + "\n"


def _split_paragraphs(text: str) -> list[str]:
    chunks = re.split(r"\n\s*\n", text.strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _chunk_paragraphs(paragraphs: list[str], max_count: int) -> list[str]:
    grouped: list[str] = []
    for index in range(0, len(paragraphs), max_count):
        grouped.append("\n\n".join(paragraphs[index : index + max_count]))
    return grouped


def _strip_hearing_date_prefix(text: str) -> tuple[str, str | None]:
    match = re.match(r"^\s*Hearing date:\s*([^.\n]+)\.\s*(.*)$", text, re.DOTALL)
    if match:
        return match.group(2).strip(), match.group(1).strip()
    return text.strip(), None


def _remove_hearing_date_mentions(text: str) -> str:
    return re.sub(
        r"Hearing date:\s*[^.\n]{3,80}\.?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def _normalize_hearing_date(value: str) -> str:
    cleaned = re.sub(
        r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+",
        "",
        value.strip(),
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def _remove_standalone_date_lines(text: str) -> str:
    if not text:
        return text
    date_patterns = (
        r"^[A-Za-z]+\s+\d{1,2},\s*\d{4}$",
        r"^[A-Za-z]+\s+\d{1,2}\s+\d{4}$",
        r"^\d{1,2}/\d{1,2}/\d{2,4}$",
        r"^Hearing date:\s*.+$",
    )
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if any(re.match(pattern, stripped, re.IGNORECASE) for pattern in date_patterns):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _infer_case_name_from_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    patterns = [
        re.compile(r"\bIn\s+re\b", re.IGNORECASE),
        re.compile(r"\bIn\s+the\s+Matter\s+of\b", re.IGNORECASE),
    ]
    for line in lines:
        if any(pattern.search(line) for pattern in patterns):
            tokens = re.findall(r"[A-Za-z0-9]+", line)
            if tokens:
                return _sanitize_case_name_tokens(tokens)
    for line in lines:
        if re.search(r"\b(vs\.?|v\.)\b", line, re.IGNORECASE):
            tokens = re.findall(r"[A-Za-z0-9]+", line)
            if tokens:
                return _sanitize_case_name_tokens(tokens)
    for line in lines:
        tokens = [token for token in re.findall(r"[A-Za-z0-9]+", line) if token.isalpha()]
        if len(tokens) >= 2:
            return _sanitize_case_name_tokens(tokens[:8])
    return ""


def _compile_raw_sections(
    entries: list[dict[str, Any]],
    label_keys: tuple[str, ...],
    text_dir: Path,
) -> str:
    sections: list[str] = []
    for entry in entries:
        label = _extract_entry_value(entry, *label_keys).strip() or "Unknown"
        start_label = _extract_entry_value(entry, "start_page", "start", "starte_page").strip()
        end_label = _extract_entry_value(entry, "end_page", "end", "endpage").strip()
        start_page = _page_number_from_label(start_label)
        end_page = _page_number_from_label(end_label)
        if start_page is None or end_page is None:
            raise ValueError("Boundary entry missing start/end page.")
        if end_page < start_page:
            raise ValueError("Boundary entry has end page before start page.")
        sections.append(f"<<<{label}>>>")
        for page in range(start_page, end_page + 1):
            page_path = text_dir / f"{page:04d}.txt"
            if not page_path.exists():
                raise FileNotFoundError(f"Missing text file {page_path.name}.")
            content = page_path.read_text(encoding="utf-8", errors="ignore")
            sections.append(content.rstrip("\n"))
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def _load_classify_basic_entries(classify_path: Path) -> list[tuple[str, str, int]]:
    entries: list[tuple[str, str, int]] = []
    if not classify_path.exists():
        return entries
    with classify_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            file_name = str(payload.get("file_name", "") or "").strip()
            page_type = str(payload.get("page_type", "") or "").strip().lower()
            if not file_name or not page_type:
                continue
            page_number = _extract_page_number(file_name)
            if page_number is None:
                continue
            entries.append((file_name, page_type, page_number))
    return entries


def _natural_sort_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name)
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def _merge_pdfs(paths: list[Path], output_path: Path) -> Path:
    writer = PdfWriter()
    for path in paths:
        reader = PdfReader(str(path))
        for page in reader.pages:
            writer.add_page(page)
    with output_path.open("wb") as handle:
        writer.write(handle)
    return output_path


def _ensure_record_prep_dirs(base_dir: Path) -> tuple[Path, Path, Path]:
    root = base_dir / "record_prep"
    text_dir = root / "text_record"
    images_dir = root / "images"
    text_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    return root, text_dir, images_dir


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
        serializable[key] = value
    try:
        CONFIG_FILE.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    except OSError:
        pass


def load_classifier_settings() -> dict[str, str]:
    config = _read_config()
    api_url = str(config.get(CONFIG_KEY_CLASSIFIER_API_URL, "") or "").strip()
    model_id = str(config.get(CONFIG_KEY_CLASSIFIER_MODEL_ID, "") or "").strip()
    api_key = str(config.get(CONFIG_KEY_CLASSIFIER_API_KEY, "") or "").strip()
    prompt = str(config.get(CONFIG_KEY_CLASSIFIER_PROMPT, DEFAULT_CLASSIFIER_PROMPT) or "").strip()
    return {
        "api_url": api_url,
        "model_id": model_id,
        "api_key": api_key,
        "prompt": prompt or DEFAULT_CLASSIFIER_PROMPT,
    }


def save_classifier_settings(api_url: str, model_id: str, api_key: str, prompt: str) -> None:
    config = _read_config()
    config[CONFIG_KEY_CLASSIFIER_API_URL] = api_url
    config[CONFIG_KEY_CLASSIFIER_MODEL_ID] = model_id
    config[CONFIG_KEY_CLASSIFIER_API_KEY] = api_key
    config[CONFIG_KEY_CLASSIFIER_PROMPT] = prompt or DEFAULT_CLASSIFIER_PROMPT
    _write_config(config)


def load_classify_dates_settings() -> dict[str, str]:
    config = _read_config()
    api_url = str(config.get(CONFIG_KEY_CLASSIFY_DATES_API_URL, "") or "").strip()
    model_id = str(config.get(CONFIG_KEY_CLASSIFY_DATES_MODEL_ID, "") or "").strip()
    api_key = str(config.get(CONFIG_KEY_CLASSIFY_DATES_API_KEY, "") or "").strip()
    prompt = str(config.get(CONFIG_KEY_CLASSIFY_DATES_PROMPT, DEFAULT_CLASSIFY_DATES_PROMPT) or "").strip()
    return {
        "api_url": api_url,
        "model_id": model_id,
        "api_key": api_key,
        "prompt": prompt or DEFAULT_CLASSIFY_DATES_PROMPT,
    }


def save_classify_dates_settings(api_url: str, model_id: str, api_key: str, prompt: str) -> None:
    config = _read_config()
    config[CONFIG_KEY_CLASSIFY_DATES_API_URL] = api_url
    config[CONFIG_KEY_CLASSIFY_DATES_MODEL_ID] = model_id
    config[CONFIG_KEY_CLASSIFY_DATES_API_KEY] = api_key
    config[CONFIG_KEY_CLASSIFY_DATES_PROMPT] = prompt or DEFAULT_CLASSIFY_DATES_PROMPT
    _write_config(config)


def load_classify_report_names_settings() -> dict[str, str]:
    config = _read_config()
    api_url = str(config.get(CONFIG_KEY_CLASSIFY_REPORTS_API_URL, "") or "").strip()
    model_id = str(config.get(CONFIG_KEY_CLASSIFY_REPORTS_MODEL_ID, "") or "").strip()
    api_key = str(config.get(CONFIG_KEY_CLASSIFY_REPORTS_API_KEY, "") or "").strip()
    prompt = str(config.get(CONFIG_KEY_CLASSIFY_REPORTS_PROMPT, DEFAULT_CLASSIFY_REPORT_NAMES_PROMPT) or "").strip()
    return {
        "api_url": api_url,
        "model_id": model_id,
        "api_key": api_key,
        "prompt": prompt or DEFAULT_CLASSIFY_REPORT_NAMES_PROMPT,
    }


def save_classify_report_names_settings(
    api_url: str,
    model_id: str,
    api_key: str,
    prompt: str,
) -> None:
    config = _read_config()
    config[CONFIG_KEY_CLASSIFY_REPORTS_API_URL] = api_url
    config[CONFIG_KEY_CLASSIFY_REPORTS_MODEL_ID] = model_id
    config[CONFIG_KEY_CLASSIFY_REPORTS_API_KEY] = api_key
    config[CONFIG_KEY_CLASSIFY_REPORTS_PROMPT] = prompt or DEFAULT_CLASSIFY_REPORT_NAMES_PROMPT
    _write_config(config)


def load_classify_form_names_settings() -> dict[str, str]:
    config = _read_config()
    api_url = str(config.get(CONFIG_KEY_CLASSIFY_FORMS_API_URL, "") or "").strip()
    model_id = str(config.get(CONFIG_KEY_CLASSIFY_FORMS_MODEL_ID, "") or "").strip()
    api_key = str(config.get(CONFIG_KEY_CLASSIFY_FORMS_API_KEY, "") or "").strip()
    prompt = str(config.get(CONFIG_KEY_CLASSIFY_FORMS_PROMPT, DEFAULT_CLASSIFY_FORM_NAMES_PROMPT) or "").strip()
    return {
        "api_url": api_url,
        "model_id": model_id,
        "api_key": api_key,
        "prompt": prompt or DEFAULT_CLASSIFY_FORM_NAMES_PROMPT,
    }


def save_classify_form_names_settings(
    api_url: str,
    model_id: str,
    api_key: str,
    prompt: str,
) -> None:
    config = _read_config()
    config[CONFIG_KEY_CLASSIFY_FORMS_API_URL] = api_url
    config[CONFIG_KEY_CLASSIFY_FORMS_MODEL_ID] = model_id
    config[CONFIG_KEY_CLASSIFY_FORMS_API_KEY] = api_key
    config[CONFIG_KEY_CLASSIFY_FORMS_PROMPT] = prompt or DEFAULT_CLASSIFY_FORM_NAMES_PROMPT
    _write_config(config)


def load_case_name_settings() -> dict[str, str]:
    config = _read_config()
    api_url = str(config.get(CONFIG_KEY_CASE_NAME_API_URL, "") or "").strip()
    model_id = str(config.get(CONFIG_KEY_CASE_NAME_MODEL_ID, "") or "").strip()
    api_key = str(config.get(CONFIG_KEY_CASE_NAME_API_KEY, "") or "").strip()
    prompt = str(config.get(CONFIG_KEY_CASE_NAME_PROMPT, DEFAULT_CASE_NAME_PROMPT) or "").strip()
    return {
        "api_url": api_url,
        "model_id": model_id,
        "api_key": api_key,
        "prompt": prompt or DEFAULT_CASE_NAME_PROMPT,
    }


def save_case_name_settings(api_url: str, model_id: str, api_key: str, prompt: str) -> None:
    config = _read_config()
    config[CONFIG_KEY_CASE_NAME_API_URL] = api_url
    config[CONFIG_KEY_CASE_NAME_MODEL_ID] = model_id
    config[CONFIG_KEY_CASE_NAME_API_KEY] = api_key
    config[CONFIG_KEY_CASE_NAME_PROMPT] = prompt or DEFAULT_CASE_NAME_PROMPT
    _write_config(config)


def load_case_context() -> tuple[str, Path | None]:
    config = _read_config()
    case_name = str(config.get(CONFIG_KEY_CASE_NAME, "") or "").strip()
    root_value = str(config.get(CONFIG_KEY_CASE_ROOT_DIR, "") or "").strip()
    root_dir = Path(root_value) if root_value else None
    if root_dir is not None and not root_dir.exists():
        root_dir = None
    return case_name, root_dir


def save_case_context(case_name: str, root_dir: Path) -> None:
    config = _read_config()
    config[CONFIG_KEY_CASE_NAME] = case_name
    config[CONFIG_KEY_CASE_ROOT_DIR] = str(root_dir)
    _write_config(config)


def load_selected_pdfs() -> list[Path]:
    config = _read_config()
    raw = config.get(CONFIG_KEY_SELECTED_PDFS)
    if isinstance(raw, list):
        paths = [Path(item) for item in raw if isinstance(item, str) and item.strip()]
        return [path for path in paths if path.exists()]
    return []


def save_selected_pdfs(paths: list[Path]) -> None:
    config = _read_config()
    config[CONFIG_KEY_SELECTED_PDFS] = [str(path) for path in paths]
    _write_config(config)

def _generate_text_files(pdf_path: Path, text_dir: Path) -> None:
    with pdf_path.open("rb") as handle:
        pdf = pdftotext.PDF(handle, physical=True)
    for index, page_text in enumerate(pdf, start=1):
        target = text_dir / f"{index:04d}.txt"
        target.write_text(page_text, encoding="utf-8")


def _generate_image_files(pdf_path: Path, images_dir: Path) -> None:
    doc = fitz.open(str(pdf_path))
    try:
        for index in range(len(doc)):
            page = doc.load_page(index)
            pix = page.get_pixmap(dpi=300, colorspace=fitz.csGRAY)
            pix.save(str(images_dir / f"{index + 1:04d}.png"))
    finally:
        doc.close()

@dataclass
class ClassifySettingsWidgets:
    api_url_row: Adw.EntryRow
    model_row: Adw.EntryRow
    api_key_row: Adw.EntryRow
    prompt_buffer: Gtk.TextBuffer


@dataclass
class OptimizeSettingsWidgets:
    api_url_row: Adw.EntryRow
    model_row: Adw.EntryRow
    api_key_row: Adw.EntryRow
    attorneys_prompt_buffer: Gtk.TextBuffer
    hearings_prompt_buffer: Gtk.TextBuffer
    reports_prompt_buffer: Gtk.TextBuffer


@dataclass
class SummarizeSettingsWidgets:
    api_url_row: Adw.EntryRow
    model_row: Adw.EntryRow
    api_key_row: Adw.EntryRow
    chunk_size_row: Adw.EntryRow
    hearings_prompt_buffer: Gtk.TextBuffer
    reports_prompt_buffer: Gtk.TextBuffer


def load_optimize_settings() -> dict[str, str]:
    config = _read_config()
    api_url = str(config.get(CONFIG_KEY_OPTIMIZE_API_URL, "") or "").strip()
    model_id = str(config.get(CONFIG_KEY_OPTIMIZE_MODEL_ID, "") or "").strip()
    api_key = str(config.get(CONFIG_KEY_OPTIMIZE_API_KEY, "") or "").strip()
    attorneys_prompt = str(
        config.get(CONFIG_KEY_OPTIMIZE_ATTORNEYS_PROMPT, DEFAULT_OPTIMIZE_ATTORNEYS_PROMPT) or ""
    ).strip()
    hearings_prompt = str(
        config.get(CONFIG_KEY_OPTIMIZE_HEARINGS_PROMPT, DEFAULT_OPTIMIZE_HEARINGS_PROMPT) or ""
    ).strip()
    reports_prompt = str(
        config.get(CONFIG_KEY_OPTIMIZE_REPORTS_PROMPT, DEFAULT_OPTIMIZE_REPORTS_PROMPT) or ""
    ).strip()
    return {
        "api_url": api_url,
        "model_id": model_id,
        "api_key": api_key,
        "attorneys_prompt": attorneys_prompt or DEFAULT_OPTIMIZE_ATTORNEYS_PROMPT,
        "hearings_prompt": hearings_prompt or DEFAULT_OPTIMIZE_HEARINGS_PROMPT,
        "reports_prompt": reports_prompt or DEFAULT_OPTIMIZE_REPORTS_PROMPT,
    }


def save_optimize_settings(
    api_url: str,
    model_id: str,
    api_key: str,
    attorneys_prompt: str,
    hearings_prompt: str,
    reports_prompt: str,
) -> None:
    config = _read_config()
    config[CONFIG_KEY_OPTIMIZE_API_URL] = api_url
    config[CONFIG_KEY_OPTIMIZE_MODEL_ID] = model_id
    config[CONFIG_KEY_OPTIMIZE_API_KEY] = api_key
    config[CONFIG_KEY_OPTIMIZE_ATTORNEYS_PROMPT] = attorneys_prompt or DEFAULT_OPTIMIZE_ATTORNEYS_PROMPT
    config[CONFIG_KEY_OPTIMIZE_HEARINGS_PROMPT] = hearings_prompt or DEFAULT_OPTIMIZE_HEARINGS_PROMPT
    config[CONFIG_KEY_OPTIMIZE_REPORTS_PROMPT] = reports_prompt or DEFAULT_OPTIMIZE_REPORTS_PROMPT
    _write_config(config)


def load_summarize_settings() -> dict[str, str]:
    config = _read_config()
    api_url = str(config.get(CONFIG_KEY_SUMMARIZE_API_URL, "") or "").strip()
    model_id = str(config.get(CONFIG_KEY_SUMMARIZE_MODEL_ID, "") or "").strip()
    api_key = str(config.get(CONFIG_KEY_SUMMARIZE_API_KEY, "") or "").strip()
    chunk_size_raw = str(config.get(CONFIG_KEY_SUMMARIZE_CHUNK_SIZE, "") or "").strip()
    chunk_size = DEFAULT_SUMMARIZE_CHUNK_SIZE
    if chunk_size_raw:
        try:
            chunk_size = max(1, int(chunk_size_raw))
        except ValueError:
            chunk_size = DEFAULT_SUMMARIZE_CHUNK_SIZE
    hearings_prompt = str(
        config.get(CONFIG_KEY_SUMMARIZE_HEARINGS_PROMPT, DEFAULT_SUMMARIZE_HEARINGS_PROMPT) or ""
    ).strip()
    reports_prompt = str(
        config.get(CONFIG_KEY_SUMMARIZE_REPORTS_PROMPT, DEFAULT_SUMMARIZE_REPORTS_PROMPT) or ""
    ).strip()
    return {
        "api_url": api_url,
        "model_id": model_id,
        "api_key": api_key,
        "chunk_size": str(chunk_size),
        "hearings_prompt": hearings_prompt or DEFAULT_SUMMARIZE_HEARINGS_PROMPT,
        "reports_prompt": reports_prompt or DEFAULT_SUMMARIZE_REPORTS_PROMPT,
    }


def save_summarize_settings(
    api_url: str,
    model_id: str,
    api_key: str,
    chunk_size: str,
    hearings_prompt: str,
    reports_prompt: str,
) -> None:
    config = _read_config()
    config[CONFIG_KEY_SUMMARIZE_API_URL] = api_url
    config[CONFIG_KEY_SUMMARIZE_MODEL_ID] = model_id
    config[CONFIG_KEY_SUMMARIZE_API_KEY] = api_key
    config[CONFIG_KEY_SUMMARIZE_CHUNK_SIZE] = chunk_size
    config[CONFIG_KEY_SUMMARIZE_HEARINGS_PROMPT] = (
        hearings_prompt or DEFAULT_SUMMARIZE_HEARINGS_PROMPT
    )
    config[CONFIG_KEY_SUMMARIZE_REPORTS_PROMPT] = (
        reports_prompt or DEFAULT_SUMMARIZE_REPORTS_PROMPT
    )
    _write_config(config)


class SettingsWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application, on_saved: Callable[[], None] | None = None) -> None:
        super().__init__(application=app, title="Settings")
        self.set_default_size(900, 720)
        self.set_resizable(True)
        self._on_saved = on_saved
        self._prompt_editors: dict[str, ClassifySettingsWidgets] = {}
        self._prompt_row_keys: dict[Gtk.ListBoxRow, str] = {}
        self._build_ui()

    def trigger_save(self) -> None:
        self._save_settings()

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
        prompt_list_scroller.set_min_content_width(220)
        prompt_list_scroller.set_child(prompt_list)

        prompt_stack = Gtk.Stack()
        prompt_stack.set_hexpand(True)
        prompt_stack.set_vexpand(True)
        prompt_stack.set_transition_type(Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)
        self._prompt_stack = prompt_stack

        prompt_definitions = [
            ("case-name", "Infer Case Name", load_case_name_settings(), DEFAULT_CASE_NAME_PROMPT),
            ("basic", "Classify Basic", load_classifier_settings(), DEFAULT_CLASSIFIER_PROMPT),
            ("dates", "Classify Dates", load_classify_dates_settings(), DEFAULT_CLASSIFY_DATES_PROMPT),
            (
                "report-names",
                "Classify Report Names",
                load_classify_report_names_settings(),
                DEFAULT_CLASSIFY_REPORT_NAMES_PROMPT,
            ),
            (
                "form-names",
                "Classify Form Names",
                load_classify_form_names_settings(),
                DEFAULT_CLASSIFY_FORM_NAMES_PROMPT,
            ),
        ]
        first_row: Gtk.ListBoxRow | None = None
        for key, title, settings, default_prompt in prompt_definitions:
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

            page = self._build_prompt_page(key, title, settings, default_prompt)
            prompt_stack.add_named(page, key)

        optimize_row = Gtk.ListBoxRow()
        optimize_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        optimize_box.set_margin_top(8)
        optimize_box.set_margin_bottom(8)
        optimize_box.set_margin_start(12)
        optimize_box.set_margin_end(12)
        optimize_label = Gtk.Label(label="Optimize Summaries", xalign=0)
        optimize_box.append(optimize_label)
        optimize_row.set_child(optimize_box)
        prompt_list.append(optimize_row)
        self._prompt_row_keys[optimize_row] = "optimize"
        optimize_page = self._build_optimize_prompt_page(load_optimize_settings())
        prompt_stack.add_named(optimize_page, "optimize")

        summarize_row = Gtk.ListBoxRow()
        summarize_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        summarize_box.set_margin_top(8)
        summarize_box.set_margin_bottom(8)
        summarize_box.set_margin_start(12)
        summarize_box.set_margin_end(12)
        summarize_label = Gtk.Label(label="Summarize", xalign=0)
        summarize_box.append(summarize_label)
        summarize_row.set_child(summarize_box)
        prompt_list.append(summarize_row)
        self._prompt_row_keys[summarize_row] = "summarize"
        summarize_page = self._build_summarize_prompt_page(load_summarize_settings())
        prompt_stack.add_named(summarize_page, "summarize")

        if first_row is not None:
            prompt_list.select_row(first_row)
            prompt_stack.set_visible_child_name(self._prompt_row_keys[first_row])

        split.set_start_child(prompt_list_scroller)
        split.set_end_child(prompt_stack)
        box.append(split)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)
        scrolled.set_child(box)
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
        save_btn.set_action_name("app.save-settings")
        buttons.append(save_btn)
        content.append(buttons)

        view.set_content(content)
        self.set_content(view)

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

    def _build_prompt_page(
        self,
        key: str,
        title: str,
        settings: dict[str, str],
        default_prompt: str,
    ) -> Gtk.Widget:
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
        api_url_row.set_text(settings.get("api_url", ""))
        credentials_group.add(api_url_row)

        model_row = Adw.EntryRow(title="Model ID (optional)")
        model_row.set_text(settings.get("model_id", ""))
        credentials_group.add(model_row)

        api_key_row = self._build_password_row("API Key")
        api_key_row.set_text(settings.get("api_key", ""))
        credentials_group.add(api_key_row)

        prompt_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        prompt_section.set_hexpand(True)
        prompt_section.set_vexpand(True)
        prompt_label = Gtk.Label(label="Prompt", xalign=0)
        prompt_label.add_css_class("dim-label")
        prompt_section.append(prompt_label)
        prompt_scroller, buffer = self._build_prompt_editor(settings.get("prompt") or default_prompt)
        prompt_section.append(prompt_scroller)
        page_box.append(prompt_section)

        page = Gtk.ScrolledWindow()
        page.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        page.set_hexpand(True)
        page.set_vexpand(True)
        page.set_child(page_box)

        self._prompt_editors[key] = ClassifySettingsWidgets(
            api_url_row=api_url_row,
            model_row=model_row,
            api_key_row=api_key_row,
            prompt_buffer=buffer,
        )
        return page

    def _build_optimize_prompt_page(self, settings: dict[str, str]) -> Gtk.Widget:
        page_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        page_box.set_margin_top(12)
        page_box.set_margin_bottom(12)
        page_box.set_margin_start(12)
        page_box.set_margin_end(12)
        page_box.set_vexpand(True)

        title_label = Gtk.Label(label="Optimize Summaries", xalign=0)
        title_label.add_css_class("title-3")
        page_box.append(title_label)

        credentials_group = Adw.PreferencesGroup(title="Credentials")
        credentials_group.add_css_class("list-stack")
        credentials_group.set_hexpand(True)
        page_box.append(credentials_group)

        api_url_row = Adw.EntryRow(title="API URL")
        api_url_row.set_text(settings.get("api_url", ""))
        credentials_group.add(api_url_row)

        model_row = Adw.EntryRow(title="Model ID")
        model_row.set_text(settings.get("model_id", ""))
        credentials_group.add(model_row)

        api_key_row = self._build_password_row("API Key")
        api_key_row.set_text(settings.get("api_key", ""))
        credentials_group.add(api_key_row)

        prompt_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        prompt_section.set_hexpand(True)
        prompt_section.set_vexpand(True)

        attorneys_label = Gtk.Label(label="Attorney Extraction Prompt", xalign=0)
        attorneys_label.add_css_class("dim-label")
        prompt_section.append(attorneys_label)
        attorneys_scroller, attorneys_buffer = self._build_prompt_editor(
            settings.get("attorneys_prompt") or DEFAULT_OPTIMIZE_ATTORNEYS_PROMPT
        )
        prompt_section.append(attorneys_scroller)

        hearings_label = Gtk.Label(label="Optimize Hearings Prompt", xalign=0)
        hearings_label.add_css_class("dim-label")
        prompt_section.append(hearings_label)
        hearings_scroller, hearings_buffer = self._build_prompt_editor(
            settings.get("hearings_prompt") or DEFAULT_OPTIMIZE_HEARINGS_PROMPT
        )
        prompt_section.append(hearings_scroller)

        reports_label = Gtk.Label(label="Optimize Reports Prompt", xalign=0)
        reports_label.add_css_class("dim-label")
        prompt_section.append(reports_label)
        reports_scroller, reports_buffer = self._build_prompt_editor(
            settings.get("reports_prompt") or DEFAULT_OPTIMIZE_REPORTS_PROMPT
        )
        prompt_section.append(reports_scroller)

        page_box.append(prompt_section)

        page = Gtk.ScrolledWindow()
        page.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        page.set_hexpand(True)
        page.set_vexpand(True)
        page.set_child(page_box)

        self._optimize_widgets = OptimizeSettingsWidgets(
            api_url_row=api_url_row,
            model_row=model_row,
            api_key_row=api_key_row,
            attorneys_prompt_buffer=attorneys_buffer,
            hearings_prompt_buffer=hearings_buffer,
            reports_prompt_buffer=reports_buffer,
        )
        return page

    def _build_summarize_prompt_page(self, settings: dict[str, str]) -> Gtk.Widget:
        page_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        page_box.set_margin_top(12)
        page_box.set_margin_bottom(12)
        page_box.set_margin_start(12)
        page_box.set_margin_end(12)
        page_box.set_vexpand(True)

        title_label = Gtk.Label(label="Summarize", xalign=0)
        title_label.add_css_class("title-3")
        page_box.append(title_label)

        credentials_group = Adw.PreferencesGroup(title="Credentials")
        credentials_group.add_css_class("list-stack")
        credentials_group.set_hexpand(True)
        page_box.append(credentials_group)

        api_url_row = Adw.EntryRow(title="API URL")
        api_url_row.set_text(settings.get("api_url", ""))
        credentials_group.add(api_url_row)

        model_row = Adw.EntryRow(title="Model ID")
        model_row.set_text(settings.get("model_id", ""))
        credentials_group.add(model_row)

        api_key_row = self._build_password_row("API Key")
        api_key_row.set_text(settings.get("api_key", ""))
        credentials_group.add(api_key_row)

        chunk_size_row = Adw.EntryRow(title="Chunk Size (paragraphs)")
        chunk_size_row.set_text(settings.get("chunk_size", str(DEFAULT_SUMMARIZE_CHUNK_SIZE)))
        credentials_group.add(chunk_size_row)

        prompt_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        prompt_section.set_hexpand(True)
        prompt_section.set_vexpand(True)

        hearings_label = Gtk.Label(label="Summarize Hearings Prompt", xalign=0)
        hearings_label.add_css_class("dim-label")
        prompt_section.append(hearings_label)
        hearings_scroller, hearings_buffer = self._build_prompt_editor(
            settings.get("hearings_prompt") or DEFAULT_SUMMARIZE_HEARINGS_PROMPT
        )
        prompt_section.append(hearings_scroller)

        reports_label = Gtk.Label(label="Summarize Reports Prompt", xalign=0)
        reports_label.add_css_class("dim-label")
        prompt_section.append(reports_label)
        reports_scroller, reports_buffer = self._build_prompt_editor(
            settings.get("reports_prompt") or DEFAULT_SUMMARIZE_REPORTS_PROMPT
        )
        prompt_section.append(reports_scroller)

        page_box.append(prompt_section)

        page = Gtk.ScrolledWindow()
        page.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        page.set_hexpand(True)
        page.set_vexpand(True)
        page.set_child(page_box)

        self._summarize_widgets = SummarizeSettingsWidgets(
            api_url_row=api_url_row,
            model_row=model_row,
            api_key_row=api_key_row,
            chunk_size_row=chunk_size_row,
            hearings_prompt_buffer=hearings_buffer,
            reports_prompt_buffer=reports_buffer,
        )
        return page

    def _prompt_text(self, buffer: Gtk.TextBuffer) -> str:
        start, end = buffer.get_bounds()
        return buffer.get_text(start, end, True)

    def _on_prompt_row_selected(self, _listbox: Gtk.ListBox, row: Gtk.ListBoxRow | None) -> None:
        if not row:
            return
        key = self._prompt_row_keys.get(row)
        if key:
            self._prompt_stack.set_visible_child_name(key)

    def _save_settings(self) -> None:
        case_widgets = self._prompt_editors.get("case-name")
        basic_widgets = self._prompt_editors.get("basic")
        dates_widgets = self._prompt_editors.get("dates")
        report_widgets = self._prompt_editors.get("report-names")
        form_widgets = self._prompt_editors.get("form-names")
        optimize_widgets = getattr(self, "_optimize_widgets", None)
        summarize_widgets = getattr(self, "_summarize_widgets", None)
        if case_widgets:
            save_case_name_settings(
                case_widgets.api_url_row.get_text().strip(),
                case_widgets.model_row.get_text().strip(),
                case_widgets.api_key_row.get_text().strip(),
                self._prompt_text(case_widgets.prompt_buffer).strip(),
            )
        if basic_widgets:
            save_classifier_settings(
                basic_widgets.api_url_row.get_text().strip(),
                basic_widgets.model_row.get_text().strip(),
                basic_widgets.api_key_row.get_text().strip(),
                self._prompt_text(basic_widgets.prompt_buffer).strip(),
            )
        if dates_widgets:
            save_classify_dates_settings(
                dates_widgets.api_url_row.get_text().strip(),
                dates_widgets.model_row.get_text().strip(),
                dates_widgets.api_key_row.get_text().strip(),
                self._prompt_text(dates_widgets.prompt_buffer).strip(),
            )
        if report_widgets:
            save_classify_report_names_settings(
                report_widgets.api_url_row.get_text().strip(),
                report_widgets.model_row.get_text().strip(),
                report_widgets.api_key_row.get_text().strip(),
                self._prompt_text(report_widgets.prompt_buffer).strip(),
            )
        if form_widgets:
            save_classify_form_names_settings(
                form_widgets.api_url_row.get_text().strip(),
                form_widgets.model_row.get_text().strip(),
                form_widgets.api_key_row.get_text().strip(),
                self._prompt_text(form_widgets.prompt_buffer).strip(),
            )
        if optimize_widgets:
            save_optimize_settings(
                optimize_widgets.api_url_row.get_text().strip(),
                optimize_widgets.model_row.get_text().strip(),
                optimize_widgets.api_key_row.get_text().strip(),
                self._prompt_text(optimize_widgets.attorneys_prompt_buffer).strip(),
                self._prompt_text(optimize_widgets.hearings_prompt_buffer).strip(),
                self._prompt_text(optimize_widgets.reports_prompt_buffer).strip(),
            )
        if summarize_widgets:
            save_summarize_settings(
                summarize_widgets.api_url_row.get_text().strip(),
                summarize_widgets.model_row.get_text().strip(),
                summarize_widgets.api_key_row.get_text().strip(),
                summarize_widgets.chunk_size_row.get_text().strip(),
                self._prompt_text(summarize_widgets.hearings_prompt_buffer).strip(),
                self._prompt_text(summarize_widgets.reports_prompt_buffer).strip(),
            )
        if self._on_saved:
            self._on_saved()
        self.close()


class RecordPrepWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application) -> None:
        super().__init__(application=app, title=APPLICATION_NAME)
        self.set_default_size(900, 600)

        self.selected_pdfs: list[Path] = []
        self._settings_window: SettingsWindow | None = None

        header_bar = Adw.HeaderBar()

        self.file_button = Gtk.Button.new_from_icon_name("list-add-symbolic")
        self.file_button.set_tooltip_text("Choose PDF(s)")
        self.file_button.add_css_class("flat")
        self.file_button.connect("clicked", self.on_choose_pdf)
        header_bar.pack_start(self.file_button)

        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.status_spinner = Gtk.Spinner()
        self.status_label = Gtk.Label(label="Idle", xalign=0)
        status_box.append(self.status_spinner)
        status_box.append(self.status_label)
        header_bar.set_title_widget(status_box)

        self.menu_button = Gtk.MenuButton(icon_name="open-menu-symbolic")
        header_bar.pack_end(self.menu_button)

        self.toast_overlay = Adw.ToastOverlay()
        toolbar_view = Adw.ToolbarView()
        toolbar_view.add_top_bar(header_bar)
        toolbar_view.set_content(self.toast_overlay)
        self.set_content(toolbar_view)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content.set_margin_top(24)
        content.set_margin_bottom(24)
        content.set_margin_start(24)
        content.set_margin_end(24)
        self.toast_overlay.set_child(content)

        self.selected_label = Gtk.Label(label="Selected: None", xalign=0)
        self.selected_label.add_css_class("dim-label")
        content.append(self.selected_label)

        listbox = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        listbox.add_css_class("boxed-list")
        content.append(listbox)

        self.step_one_row = Adw.ActionRow(
            title="Create files",
            subtitle="Generate per-page text and image files for the selected PDFs.",
        )
        self.step_one_row.set_activatable(True)
        self.step_one_row.connect("activated", self.on_step_one_clicked)
        listbox.append(self.step_one_row)

        self.step_strip_nonstandard_row = Adw.ActionRow(
            title="Strip characters",
            subtitle="Remove non-printing characters from the extracted text files.",
        )
        self.step_strip_nonstandard_row.set_activatable(True)
        self.step_strip_nonstandard_row.connect("activated", self.on_step_strip_nonstandard_clicked)
        listbox.append(self.step_strip_nonstandard_row)

        self.step_infer_case_row = Adw.ActionRow(
            title="Infer case",
            subtitle="Use the first pages to infer the case name and save it.",
        )
        self.step_infer_case_row.set_activatable(True)
        self.step_infer_case_row.connect("activated", self.on_step_infer_case_clicked)
        listbox.append(self.step_infer_case_row)

        self.step_two_row = Adw.ActionRow(
            title="Classify pages",
            subtitle="Label each page by type, date, and form metadata.",
        )
        self.step_two_row.set_activatable(True)
        self.step_two_row.connect("activated", self.on_step_two_clicked)
        listbox.append(self.step_two_row)

        self.step_three_row = Adw.ActionRow(
            title="Classify dates",
            subtitle="Identify hearing and minute-order dates from the transcript.",
        )
        self.step_three_row.set_activatable(True)
        self.step_three_row.connect("activated", self.on_step_three_clicked)
        listbox.append(self.step_three_row)

        self.step_four_row = Adw.ActionRow(
            title="Classify reports",
            subtitle="Extract report titles from report sections.",
        )
        self.step_four_row.set_activatable(True)
        self.step_four_row.connect("activated", self.on_step_four_clicked)
        listbox.append(self.step_four_row)

        self.step_five_row = Adw.ActionRow(
            title="Classify forms",
            subtitle="Extract form names from form pages.",
        )
        self.step_five_row.set_activatable(True)
        self.step_five_row.connect("activated", self.on_step_five_clicked)
        listbox.append(self.step_five_row)

        self.step_six_row = Adw.ActionRow(
            title="Build TOC",
            subtitle="Compile a table of contents for forms, reports, orders, and hearings.",
        )
        self.step_six_row.set_activatable(True)
        self.step_six_row.connect("activated", self.on_step_six_clicked)
        listbox.append(self.step_six_row)

        self.step_seven_row = Adw.ActionRow(
            title="Find boundaries",
            subtitle="Determine page ranges for hearing and report sections.",
        )
        self.step_seven_row.set_activatable(True)
        self.step_seven_row.connect("activated", self.on_step_seven_clicked)
        listbox.append(self.step_seven_row)

        self.step_eight_row = Adw.ActionRow(
            title="Create raw",
            subtitle="Create raw hearing and report text files for summarization.",
        )
        self.step_eight_row.set_activatable(True)
        self.step_eight_row.connect("activated", self.on_step_eight_clicked)
        listbox.append(self.step_eight_row)

        self.step_nine_row = Adw.ActionRow(
            title="Create optimized",
            subtitle="Prepare optimized hearing and report text for retrieval.",
        )
        self.step_nine_row.set_activatable(True)
        self.step_nine_row.connect("activated", self.on_step_nine_clicked)
        listbox.append(self.step_nine_row)

        self.step_ten_row = Adw.ActionRow(
            title="Create summaries",
            subtitle="Summarize optimized hearings and reports into concise paragraphs.",
        )
        self.step_ten_row.set_activatable(True)
        self.step_ten_row.connect("activated", self.on_step_ten_clicked)
        listbox.append(self.step_ten_row)

        self._setup_menu(app)
        self._load_selected_pdfs()
        self._load_case_context()
        self._set_status(APPLICATION_NAME, False)

    def _setup_menu(self, app: Adw.Application) -> None:
        menu = Gio.Menu()
        menu.append("Settings", "app.settings")
        self.menu_button.set_menu_model(menu)

        action = Gio.SimpleAction.new("settings", None)
        action.connect("activate", self.on_settings)
        app.add_action(action)

        if app.lookup_action("save-settings") is None:
            save_action = Gio.SimpleAction.new("save-settings", None)
            save_action.connect("activate", self._on_action_save_settings)
            app.add_action(save_action)

    def on_settings(self, _action: Gio.SimpleAction, _param: object) -> None:
        if self._settings_window:
            self._settings_window.present()
            return
        settings = SettingsWindow(self.get_application(), on_saved=self._on_settings_saved)
        settings.connect("close-request", self._on_settings_close_request)
        self._settings_window = settings
        settings.present()

    def _on_settings_saved(self) -> None:
        self.show_toast("Settings saved.")

    def _on_settings_close_request(self, _window: SettingsWindow) -> bool:
        self._settings_window = None
        return False

    def _on_action_save_settings(self, _action: Gio.SimpleAction, _param: object) -> None:
        if not self._settings_window:
            self.show_toast("Settings window is not open.")
            return
        self._settings_window.trigger_save()

    def show_toast(self, message: str) -> None:
        self.toast_overlay.add_toast(Adw.Toast(title=message))

    def _set_status(self, message: str, active: bool) -> None:
        self.status_label.set_text(message)
        if active:
            self.status_spinner.start()
        else:
            self.status_spinner.stop()

    def _start_step(self, row: Adw.ActionRow) -> None:
        title = row.get_title() or "Working"
        self._set_status(f"Working: {title}", True)

    def _stop_status(self) -> None:
        self._set_status(APPLICATION_NAME, False)

    def on_choose_pdf(self, _button: Gtk.Button) -> None:
        dialog = Gtk.FileDialog(title="Choose PDF files")
        file_filter = Gtk.FileFilter()
        file_filter.add_mime_type("application/pdf")
        file_filter.set_name("PDF files")
        dialog.set_default_filter(file_filter)
        dialog.open_multiple(self, None, self._on_files_chosen)

    def _on_files_chosen(
        self, dialog: Gtk.FileDialog, result: Gio.AsyncResult
    ) -> None:
        try:
            files = dialog.open_multiple_finish(result)
        except GLib.Error:
            return
        paths: list[Path] = []
        for index in range(files.get_n_items()):
            gfile = files.get_item(index)
            if not isinstance(gfile, Gio.File):
                continue
            path = gfile.get_path()
            if path:
                paths.append(Path(path))
        if not paths:
            self.show_toast("No PDFs selected.")
            return
        self.selected_pdfs = sorted(paths, key=_natural_sort_key)
        save_selected_pdfs(self.selected_pdfs)
        label = (
            self.selected_pdfs[0].name
            if len(self.selected_pdfs) == 1
            else f"{len(self.selected_pdfs)} PDFs selected"
        )
        self.selected_label.set_text(f"Selected: {label}")
        self.show_toast(f"Selected: {label}")

    def _load_selected_pdfs(self) -> None:
        self.selected_pdfs = load_selected_pdfs()
        if not self.selected_pdfs:
            return
        label = (
            self.selected_pdfs[0].name
            if len(self.selected_pdfs) == 1
            else f"{len(self.selected_pdfs)} PDFs selected"
        )
        self.selected_label.set_text(f"Selected: {label}")

    def _load_case_context(self) -> None:
        case_name, _root_dir = load_case_context()
        if case_name:
            self.selected_label.set_text(f"Selected: {case_name}")

    def on_step_one_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self.step_one_row.set_sensitive(False)
        self._start_step(self.step_one_row)
        threading.Thread(target=self._run_step_one, daemon=True).start()

    def on_step_strip_nonstandard_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self.step_strip_nonstandard_row.set_sensitive(False)
        self._start_step(self.step_strip_nonstandard_row)
        threading.Thread(target=self._run_step_strip_nonstandard, daemon=True).start()

    def on_step_infer_case_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self.step_infer_case_row.set_sensitive(False)
        self._start_step(self.step_infer_case_row)
        threading.Thread(target=self._run_step_infer_case, daemon=True).start()

    def _run_step_one(self) -> None:
        try:
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir, text_dir, images_dir = _ensure_record_prep_dirs(base_dir)
            if len(self.selected_pdfs) > 1:
                merged_path = root_dir / "merged.pdf"
                pdf_path = _merge_pdfs(self.selected_pdfs, merged_path)
            else:
                pdf_path = self.selected_pdfs[0]
            _generate_text_files(pdf_path, text_dir)
            _generate_image_files(pdf_path, images_dir)
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Step 1 failed: {exc}")
        else:
            GLib.idle_add(self.show_toast, "Step 1 complete.")
        finally:
            GLib.idle_add(self.step_one_row.set_sensitive, True)
            GLib.idle_add(self._stop_status)

    def _run_step_strip_nonstandard(self) -> None:
        try:
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "record_prep"
            text_dir = root_dir / "text_record"
            if not text_dir.exists():
                raise FileNotFoundError("Run Step 1 to generate text files first.")
            text_files = sorted(text_dir.glob("*.txt"), key=_natural_sort_key)
            if not text_files:
                raise FileNotFoundError("No text files found to sanitize.")
            for text_path in text_files:
                content = text_path.read_text(encoding="utf-8", errors="ignore")
                cleaned = _strip_nonstandard_characters(content)
                if cleaned != content:
                    text_path.write_text(cleaned, encoding="utf-8")
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Strip non-standard characters failed: {exc}")
        else:
            GLib.idle_add(self.show_toast, "Strip non-standard characters complete.")
        finally:
            GLib.idle_add(self.step_strip_nonstandard_row.set_sensitive, True)
            GLib.idle_add(self._stop_status)

    def _run_step_infer_case(self) -> None:
        try:
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "record_prep"
            text_dir = root_dir / "text_record"
            if not text_dir.exists():
                raise FileNotFoundError("Run Step 1 to generate text files first.")
            settings = load_case_name_settings()
            if not settings["api_url"] or not settings["model_id"] or not settings["api_key"]:
                raise ValueError("Configure case name API URL, model ID, and API key in Settings.")
            text_files = sorted(text_dir.glob("*.txt"), key=_natural_sort_key)[:3]
            if not text_files:
                raise FileNotFoundError("No text files found to infer case name.")
            combined = "\n".join(
                path.read_text(encoding="utf-8", errors="ignore") for path in text_files
            )
            response_text = self._request_plain_text(settings, combined)
            case_name = _sanitize_case_name_value(response_text)
            if not case_name:
                case_name = _sanitize_case_name_value(_infer_case_name_from_text(combined))
            if not case_name:
                raise ValueError("Unable to infer case name from first three pages.")
            (root_dir / "case_name.txt").write_text(case_name, encoding="utf-8")
            save_case_context(case_name, base_dir)
            GLib.idle_add(self.selected_label.set_text, f"Selected: {case_name}")
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Infer case name failed: {exc}")
        else:
            GLib.idle_add(self.show_toast, "Infer case name complete.")
        finally:
            GLib.idle_add(self.step_infer_case_row.set_sensitive, True)
            GLib.idle_add(self._stop_status)

    def on_step_two_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self.step_two_row.set_sensitive(False)
        self._start_step(self.step_two_row)
        threading.Thread(target=self._run_step_two, daemon=True).start()

    def on_step_three_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self.step_three_row.set_sensitive(False)
        self._start_step(self.step_three_row)
        threading.Thread(target=self._run_step_three, daemon=True).start()

    def on_step_four_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self.step_four_row.set_sensitive(False)
        self._start_step(self.step_four_row)
        threading.Thread(target=self._run_step_four, daemon=True).start()

    def on_step_five_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self.step_five_row.set_sensitive(False)
        self._start_step(self.step_five_row)
        threading.Thread(target=self._run_step_five, daemon=True).start()

    def on_step_six_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self.step_six_row.set_sensitive(False)
        self._start_step(self.step_six_row)
        threading.Thread(target=self._run_step_six, daemon=True).start()

    def on_step_seven_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self.step_seven_row.set_sensitive(False)
        self._start_step(self.step_seven_row)
        threading.Thread(target=self._run_step_seven, daemon=True).start()

    def on_step_eight_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self.step_eight_row.set_sensitive(False)
        self._start_step(self.step_eight_row)
        threading.Thread(target=self._run_step_eight, daemon=True).start()

    def on_step_nine_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self.step_nine_row.set_sensitive(False)
        self._start_step(self.step_nine_row)
        threading.Thread(target=self._run_step_nine, daemon=True).start()

    def on_step_ten_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self.step_ten_row.set_sensitive(False)
        self._start_step(self.step_ten_row)
        threading.Thread(target=self._run_step_ten, daemon=True).start()

    def _run_step_two(self) -> None:
        try:
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "record_prep"
            text_dir = root_dir / "text_record"
            if not text_dir.exists():
                raise FileNotFoundError("Run Step 1 to generate text files first.")
            settings = load_classifier_settings()
            if not settings["api_url"] or not settings["model_id"] or not settings["api_key"]:
                raise ValueError("Configure classifier API URL, model ID, and API key in Settings.")
            classification_dir = root_dir / "classification"
            classification_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = classification_dir / "classify_basic.jsonl"
            jsonl_path.write_text("", encoding="utf-8")
            text_files = sorted(text_dir.glob("*.txt"), key=_natural_sort_key)
            if not text_files:
                raise FileNotFoundError("No text files found to classify.")
            for text_path in text_files:
                content = text_path.read_text(encoding="utf-8", errors="ignore")
                entry = self._classify_text(settings, text_path.name, content)
                with jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry))
                    handle.write("\n")
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Step 2 failed: {exc}")
        else:
            GLib.idle_add(self.show_toast, "Step 2 complete.")
        finally:
            GLib.idle_add(self.step_two_row.set_sensitive, True)
            GLib.idle_add(self._stop_status)

    def _run_step_three(self) -> None:
        try:
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "record_prep"
            text_dir = root_dir / "text_record"
            if not text_dir.exists():
                raise FileNotFoundError("Run Step 1 to generate text files first.")
            classification_dir = root_dir / "classification"
            classify_basic_path = classification_dir / "classify_basic.jsonl"
            if not classify_basic_path.exists():
                raise FileNotFoundError("Run Step 2 to generate classify_basic.jsonl first.")
            settings = load_classify_dates_settings()
            if not settings["api_url"] or not settings["model_id"] or not settings["api_key"]:
                raise ValueError("Configure classify dates API URL, model ID, and API key in Settings.")
            targets = _load_classify_date_targets(classify_basic_path)
            if not targets:
                raise FileNotFoundError("No hearing or minute_order sequences found in classify_basic.jsonl.")
            classification_dir = root_dir / "classification"
            classification_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = classification_dir / "classify_dates.jsonl"
            jsonl_path.write_text("", encoding="utf-8")
            for file_name, _page_type in targets:
                text_path = text_dir / file_name
                if not text_path.exists():
                    raise FileNotFoundError(f"Missing text file {file_name} for date classification.")
                content = text_path.read_text(encoding="utf-8", errors="ignore")
                entry = self._classify_text(settings, file_name, content)
                ordered_entry: dict[str, str] = {"file_name": file_name}
                for key, value in entry.items():
                    if key == "file_name":
                        continue
                    ordered_entry[key] = value
                with jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(ordered_entry))
                    handle.write("\n")
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Step 3 failed: {exc}")
        else:
            GLib.idle_add(self.show_toast, "Step 3 complete.")
        finally:
            GLib.idle_add(self.step_three_row.set_sensitive, True)
            GLib.idle_add(self._stop_status)

    def _run_step_four(self) -> None:
        try:
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "record_prep"
            text_dir = root_dir / "text_record"
            if not text_dir.exists():
                raise FileNotFoundError("Run Step 1 to generate text files first.")
            classification_dir = root_dir / "classification"
            classify_basic_path = classification_dir / "classify_basic.jsonl"
            if not classify_basic_path.exists():
                raise FileNotFoundError("Run Step 2 to generate classify_basic.jsonl first.")
            settings = load_classify_report_names_settings()
            if not settings["api_url"] or not settings["model_id"] or not settings["api_key"]:
                raise ValueError("Configure report names API URL, model ID, and API key in Settings.")
            targets = _load_classify_report_targets(classify_basic_path)
            if not targets:
                raise FileNotFoundError("No report sequences found in classify_basic.jsonl.")
            classification_dir = root_dir / "classification"
            classification_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = classification_dir / "classify_report_names.jsonl"
            jsonl_path.write_text("", encoding="utf-8")
            for file_name, _page_type in targets:
                text_path = text_dir / file_name
                if not text_path.exists():
                    raise FileNotFoundError(f"Missing text file {file_name} for report name classification.")
                content = text_path.read_text(encoding="utf-8", errors="ignore")
                entry = self._classify_text(settings, file_name, content)
                ordered_entry: dict[str, str] = {"file_name": file_name}
                for key, value in entry.items():
                    if key == "file_name":
                        continue
                    ordered_entry[key] = value
                with jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(ordered_entry))
                    handle.write("\n")
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Step 4 failed: {exc}")
        else:
            GLib.idle_add(self.show_toast, "Step 4 complete.")
        finally:
            GLib.idle_add(self.step_four_row.set_sensitive, True)
            GLib.idle_add(self._stop_status)

    def _run_step_five(self) -> None:
        try:
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "record_prep"
            text_dir = root_dir / "text_record"
            if not text_dir.exists():
                raise FileNotFoundError("Run Step 1 to generate text files first.")
            classification_dir = root_dir / "classification"
            classify_basic_path = classification_dir / "classify_basic.jsonl"
            if not classify_basic_path.exists():
                raise FileNotFoundError("Run Step 2 to generate classify_basic.jsonl first.")
            settings = load_classify_form_names_settings()
            if not settings["api_url"] or not settings["model_id"] or not settings["api_key"]:
                raise ValueError("Configure form names API URL, model ID, and API key in Settings.")
            targets = _load_classify_form_targets(classify_basic_path)
            if not targets:
                raise FileNotFoundError("No form pages found in classify_basic.jsonl.")
            classification_dir = root_dir / "classification"
            classification_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = classification_dir / "classify_form_names.jsonl"
            jsonl_path.write_text("", encoding="utf-8")
            for file_name, _page_type in targets:
                text_path = text_dir / file_name
                if not text_path.exists():
                    raise FileNotFoundError(f"Missing text file {file_name} for form name classification.")
                content = text_path.read_text(encoding="utf-8", errors="ignore")
                entry = self._classify_text(settings, file_name, content)
                ordered_entry: dict[str, str] = {"file_name": file_name}
                for key, value in entry.items():
                    if key == "file_name":
                        continue
                    ordered_entry[key] = value
                with jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(ordered_entry))
                    handle.write("\n")
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Step 5 failed: {exc}")
        else:
            GLib.idle_add(self.show_toast, "Step 5 complete.")
        finally:
            GLib.idle_add(self.step_five_row.set_sensitive, True)
            GLib.idle_add(self._stop_status)

    def _run_step_six(self) -> None:
        try:
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "record_prep"
            classification_dir = root_dir / "classification"
            derived_dir = root_dir / "derived"
            classify_basic_path = classification_dir / "classify_basic.jsonl"
            classify_dates_path = classification_dir / "classify_dates.jsonl"
            classify_report_names_path = classification_dir / "classify_report_names.jsonl"
            classify_form_names_path = classification_dir / "classify_form_names.jsonl"
            for path in (
                classify_basic_path,
                classify_dates_path,
                classify_report_names_path,
                classify_form_names_path,
            ):
                if not path.exists():
                    raise FileNotFoundError("Run Steps 2-5 to generate classify JSONL files first.")
            derived_dir.mkdir(parents=True, exist_ok=True)
            date_entries = _load_jsonl_entries(classify_dates_path)
            basic_entries = _load_jsonl_entries(classify_basic_path)
            date_by_file: dict[str, str] = {}
            for entry in date_entries:
                file_name = _extract_entry_value(entry, "file_name", "filename")
                if not file_name:
                    continue
                date_value = _extract_entry_value(entry, "date")
                date_by_file[file_name] = date_value
            for entry in basic_entries:
                file_name = _extract_entry_value(entry, "file_name", "filename")
                if not file_name or file_name in date_by_file:
                    continue
                date_value = _extract_entry_value(entry, "date")
                if date_value:
                    date_by_file[file_name] = date_value
            form_lines: list[str] = []
            for entry in _load_jsonl_entries(classify_form_names_path):
                form_name = _extract_entry_value(entry, "form_name", "form")
                if not form_name:
                    continue
                page = _page_label_from_filename(
                    _extract_entry_value(entry, "file_name", "filename")
                )
                form_lines.append(_format_toc_line(form_name, page))
            report_lines: list[str] = []
            for entry in _load_jsonl_entries(classify_report_names_path):
                report_name = _extract_entry_value(entry, "report_name", "report", "name")
                if not report_name:
                    continue
                page = _page_label_from_filename(
                    _extract_entry_value(entry, "file_name", "filename")
                )
                report_lines.append(_format_toc_line(report_name, page))
            minute_order_lines: list[str] = []
            hearing_lines: list[str] = []
            for file_name, page_type in _load_classify_date_targets(classify_basic_path):
                date_value = date_by_file.get(file_name, "").strip()
                page = _page_label_from_filename(file_name)
                line = _format_toc_line(date_value, page)
                if page_type == "minute_order":
                    minute_order_lines.append(line)
                elif page_type == "hearing":
                    hearing_lines.append(line)
            toc_lines: list[str] = [
                "FORMS",
                *form_lines,
                "",
                "REPORTS",
                *report_lines,
                "",
                "MINUTE ORDERS",
                *minute_order_lines,
                "",
                "HEARINGS",
                *hearing_lines,
            ]
            toc_path = derived_dir / "TOC.txt"
            toc_path.write_text("\n".join(toc_lines).rstrip() + "\n", encoding="utf-8")
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Step 6 failed: {exc}")
        else:
            GLib.idle_add(self.show_toast, "Step 6 complete.")
        finally:
            GLib.idle_add(self.step_six_row.set_sensitive, True)
            GLib.idle_add(self._stop_status)

    def _run_step_seven(self) -> None:
        try:
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "record_prep"
            classification_dir = root_dir / "classification"
            derived_dir = root_dir / "derived"
            classify_basic_path = classification_dir / "classify_basic.jsonl"
            classify_dates_path = classification_dir / "classify_dates.jsonl"
            classify_report_names_path = classification_dir / "classify_report_names.jsonl"
            for path in (classify_basic_path, classify_dates_path, classify_report_names_path):
                if not path.exists():
                    raise FileNotFoundError("Run Steps 2-4 to generate classify JSONL files first.")
            derived_dir.mkdir(parents=True, exist_ok=True)
            date_by_file: dict[str, str] = {}
            for entry in _load_jsonl_entries(classify_dates_path):
                file_name = _extract_entry_value(entry, "file_name", "filename")
                if not file_name:
                    continue
                date_value = _extract_entry_value(entry, "date")
                if date_value:
                    date_by_file[file_name] = date_value
            report_name_by_file: dict[str, str] = {}
            for entry in _load_jsonl_entries(classify_report_names_path):
                file_name = _extract_entry_value(entry, "file_name", "filename")
                report_name = _extract_entry_value(entry, "report_name", "report", "name")
                if file_name and report_name:
                    report_name_by_file[file_name] = report_name
            hearing_boundaries: list[dict[str, str]] = []
            report_boundaries: list[dict[str, str]] = []
            entries = _load_classify_basic_entries(classify_basic_path)
            if not entries:
                raise FileNotFoundError("No entries found in classify_basic.jsonl.")
            current_type: str | None = None
            current_start_file: str | None = None
            current_end_file: str | None = None
            for file_name, page_type, page_number in entries:
                is_sequence_type = page_type in {"hearing", "report"}
                if not is_sequence_type:
                    if current_type:
                        self._append_boundary_entry(
                            current_type,
                            current_start_file,
                            current_end_file,
                            date_by_file,
                            report_name_by_file,
                            hearing_boundaries,
                            report_boundaries,
                        )
                        current_type = None
                        current_start_file = None
                        current_end_file = None
                    continue
                if (
                    page_type != current_type
                    or current_end_file is None
                    or _extract_page_number(current_end_file) != page_number - 1
                ):
                    if current_type:
                        self._append_boundary_entry(
                            current_type,
                            current_start_file,
                            current_end_file,
                            date_by_file,
                            report_name_by_file,
                            hearing_boundaries,
                            report_boundaries,
                        )
                    current_type = page_type
                    current_start_file = file_name
                    current_end_file = file_name
                else:
                    current_end_file = file_name
            if current_type:
                self._append_boundary_entry(
                    current_type,
                    current_start_file,
                    current_end_file,
                    date_by_file,
                    report_name_by_file,
                    hearing_boundaries,
                    report_boundaries,
                )
            hearing_path = derived_dir / "hearing_boundaries.json"
            hearing_path.write_text(
                json.dumps(hearing_boundaries, indent=2),
                encoding="utf-8",
            )
            report_path = derived_dir / "report_boudaries.json"
            report_path.write_text(
                json.dumps(report_boundaries, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Step 7 failed: {exc}")
        else:
            GLib.idle_add(self.show_toast, "Step 7 complete.")
        finally:
            GLib.idle_add(self.step_seven_row.set_sensitive, True)
            GLib.idle_add(self._stop_status)

    def _run_step_eight(self) -> None:
        try:
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "record_prep"
            derived_dir = root_dir / "derived"
            text_dir = root_dir / "text_record"
            if not text_dir.exists():
                raise FileNotFoundError("Run Step 1 to generate text files first.")
            hearing_path = derived_dir / "hearing_boundaries.json"
            report_path = derived_dir / "report_boudaries.json"
            if not hearing_path.exists() or not report_path.exists():
                raise FileNotFoundError("Run Step 7 to generate boundary JSON files first.")
            summarization_dir = root_dir / "summarization"
            summarization_dir.mkdir(parents=True, exist_ok=True)
            hearing_entries = _load_json_entries(hearing_path)
            report_entries = _load_json_entries(report_path)
            if not hearing_entries and not report_entries:
                raise FileNotFoundError("No hearing/report boundaries found.")
            raw_hearings = _compile_raw_sections(hearing_entries, ("date",), text_dir)
            raw_reports = _compile_raw_sections(
                report_entries,
                ("report_name", "report", "name"),
                text_dir,
            )
            (summarization_dir / "raw_hearings.txt").write_text(raw_hearings, encoding="utf-8")
            (summarization_dir / "raw_reports.txt").write_text(raw_reports, encoding="utf-8")
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Step 8 failed: {exc}")
        else:
            GLib.idle_add(self.show_toast, "Step 8 complete.")
        finally:
            GLib.idle_add(self.step_eight_row.set_sensitive, True)
            GLib.idle_add(self._stop_status)

    def _run_step_nine(self) -> None:
        try:
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "record_prep"
            summarization_dir = root_dir / "summarization"
            raw_hearings_path = summarization_dir / "raw_hearings.txt"
            raw_reports_path = summarization_dir / "raw_reports.txt"
            if not raw_hearings_path.exists() or not raw_reports_path.exists():
                raise FileNotFoundError("Run Step 8 to generate raw hearing/report files first.")
            settings = load_optimize_settings()
            if not settings["api_url"] or not settings["model_id"] or not settings["api_key"]:
                raise ValueError("Configure optimize API URL, model ID, and API key in Settings.")
            attorneys_prompt = settings["attorneys_prompt"]
            hearings_prompt = settings["hearings_prompt"]
            reports_prompt = settings["reports_prompt"]

            hearing_sections = _split_tagged_sections(raw_hearings_path.read_text(encoding="utf-8"))
            report_sections = _split_tagged_sections(raw_reports_path.read_text(encoding="utf-8"))
            if not hearing_sections and not report_sections:
                raise FileNotFoundError("No hearing/report sections found in raw files.")

            optimized_hearings: list[str] = []
            for label, content in hearing_sections:
                if not content:
                    continue
                sentences = _split_into_sentences(content)
                if not sentences:
                    continue
                attorney_excerpt = _chunk_sentences(sentences, 2000)[0]
                attorney_info = self._request_plain_text(
                    {
                        "api_url": settings["api_url"],
                        "model_id": settings["model_id"],
                        "api_key": settings["api_key"],
                        "prompt": attorneys_prompt,
                    },
                    attorney_excerpt,
                )
                chunks = _chunk_sentences(sentences, 3500)
                for chunk in chunks:
                    payload = f"Hearing date: {label}\nAttorney info: {attorney_info}\nTranscript:\n{chunk}"
                    response = self._request_plain_text(
                        {
                            "api_url": settings["api_url"],
                            "model_id": settings["model_id"],
                            "api_key": settings["api_key"],
                            "prompt": hearings_prompt,
                        },
                        payload,
                    )
                    if response:
                        optimized_hearings.append(response.strip())

            optimized_reports: list[str] = []
            for _label, content in report_sections:
                if not content:
                    continue
                sentences = _split_into_sentences(content)
                if not sentences:
                    continue
                chunks = _chunk_sentences(sentences, 3500)
                for chunk in chunks:
                    response = self._request_plain_text(
                        {
                            "api_url": settings["api_url"],
                            "model_id": settings["model_id"],
                            "api_key": settings["api_key"],
                            "prompt": reports_prompt,
                        },
                        chunk,
                    )
                    if response:
                        optimized_reports.append(response.strip())

            (summarization_dir / "optimized_hearings.txt").write_text(
                _collapse_blank_lines("\n\n".join(optimized_hearings)),
                encoding="utf-8",
            )
            (summarization_dir / "optimized_reports.txt").write_text(
                _collapse_blank_lines("\n\n".join(optimized_reports)),
                encoding="utf-8",
            )
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Step 9 failed: {exc}")
        else:
            GLib.idle_add(self.show_toast, "Step 9 complete.")
        finally:
            GLib.idle_add(self.step_nine_row.set_sensitive, True)
            GLib.idle_add(self._stop_status)

    def _run_step_ten(self) -> None:
        try:
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "record_prep"
            summarization_dir = root_dir / "summarization"
            optimized_hearings_path = summarization_dir / "optimized_hearings.txt"
            optimized_reports_path = summarization_dir / "optimized_reports.txt"
            if not optimized_hearings_path.exists() or not optimized_reports_path.exists():
                raise FileNotFoundError("Run Step 9 to generate optimized files first.")
            settings = load_summarize_settings()
            if not settings["api_url"] or not settings["model_id"] or not settings["api_key"]:
                raise ValueError("Configure summarize API URL, model ID, and API key in Settings.")
            chunk_size = DEFAULT_SUMMARIZE_CHUNK_SIZE
            chunk_size_raw = settings.get("chunk_size", "")
            if chunk_size_raw:
                try:
                    chunk_size = max(1, int(chunk_size_raw))
                except ValueError:
                    chunk_size = DEFAULT_SUMMARIZE_CHUNK_SIZE

            case_name, _root_dir = load_case_context()
            summary_hearings: list[str] = []
            summary_reports: list[str] = []

            if case_name:
                summary_hearings.extend(["RT Summary", case_name, ""])
            else:
                summary_hearings.append("RT Summary")

            hearing_paragraphs = _split_paragraphs(
                optimized_hearings_path.read_text(encoding="utf-8")
            )
            hearing_groups: list[tuple[str, list[str]]] = []
            current_date: str | None = None
            for paragraph in hearing_paragraphs:
                cleaned, date_value = _strip_hearing_date_prefix(paragraph)
                if date_value:
                    date_value = _normalize_hearing_date(date_value)
                    if current_date != date_value:
                        hearing_groups.append((date_value, []))
                        current_date = date_value
                if not hearing_groups:
                    hearing_groups.append(("HEARING", []))
                hearing_groups[-1][1].append(
                    _remove_standalone_date_lines(_remove_hearing_date_mentions(cleaned))
                )

            first_section = True
            for date_value, paragraphs in hearing_groups:
                if not first_section:
                    summary_hearings.append("")
                summary_hearings.append(date_value or "HEARING")
                summary_hearings.append("")
                first_section = False
                for chunk in _chunk_paragraphs(paragraphs, chunk_size):
                    response = self._request_plain_text(
                        {
                            "api_url": settings["api_url"],
                            "model_id": settings["model_id"],
                            "api_key": settings["api_key"],
                            "prompt": settings["hearings_prompt"],
                        },
                        chunk,
                    )
                    if response:
                        cleaned_response = _remove_hearing_date_mentions(response.strip())
                        summary_hearings.append(_remove_standalone_date_lines(cleaned_response))
                        summary_hearings.append("")

            if case_name:
                summary_reports.extend(["CT Summary", case_name, "", ""])
            else:
                summary_reports.extend(["CT Summary", ""])

            report_paragraphs = _split_paragraphs(
                optimized_reports_path.read_text(encoding="utf-8")
            )
            report_paragraphs = [
                re.sub(r"^Reporting:\s*", "", paragraph) for paragraph in report_paragraphs
            ]
            for chunk in _chunk_paragraphs(report_paragraphs, chunk_size):
                response = self._request_plain_text(
                    {
                        "api_url": settings["api_url"],
                        "model_id": settings["model_id"],
                        "api_key": settings["api_key"],
                        "prompt": settings["reports_prompt"],
                    },
                    chunk,
                )
                if response:
                    summary_reports.append(response.strip())
                    summary_reports.append("")

            (summarization_dir / "summarized_hearings.txt").write_text(
                _collapse_blank_lines("\n".join(summary_hearings)),
                encoding="utf-8",
            )
            (summarization_dir / "summarized_reports.txt").write_text(
                _collapse_blank_lines("\n".join(summary_reports)),
                encoding="utf-8",
            )
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Step 10 failed: {exc}")
        else:
            GLib.idle_add(self.show_toast, "Step 10 complete.")
        finally:
            GLib.idle_add(self.step_ten_row.set_sensitive, True)
            GLib.idle_add(self._stop_status)

    def _append_boundary_entry(
        self,
        page_type: str | None,
        start_file: str | None,
        end_file: str | None,
        date_by_file: dict[str, str],
        report_name_by_file: dict[str, str],
        hearing_boundaries: list[dict[str, str]],
        report_boundaries: list[dict[str, str]],
    ) -> None:
        if not page_type or not start_file or not end_file:
            return
        start_page = _page_label_from_filename(start_file)
        end_page = _page_label_from_filename(end_file)
        if page_type == "hearing":
            hearing_boundaries.append(
                {
                    "date": date_by_file.get(start_file, ""),
                    "start_page": start_page,
                    "end_page": end_page,
                }
            )
            return
        if page_type == "report":
            report_boundaries.append(
                {
                    "report_name": report_name_by_file.get(start_file, ""),
                    "start_page": start_page,
                    "end_page": end_page,
                }
            )

    def _classify_text(
        self,
        settings: dict[str, str],
        filename: str,
        content: str,
    ) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {settings['api_key']}",
            "User-Agent": "RecordPrep/0.1",
        }
        body = {
            "model": settings["model_id"],
            "stream": False,
            "messages": [
                {"role": "system", "content": settings["prompt"]},
                {"role": "user", "content": content},
            ],
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(settings["api_url"], data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
        except urllib.error.HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                error_body = ""
            detail = error_body.strip() or exc.reason or "request failed"
            raise RuntimeError(f"Classifier request failed: HTTP {exc.code} {detail}") from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Classifier request failed: {exc}") from exc
        response_text = self._extract_response_text(payload)
        try:
            parsed = json.loads(self._extract_json_payload(response_text))
        except json.JSONDecodeError:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        expected_keys = _extract_prompt_keys(settings.get("prompt", ""))
        filename_key = "file_name"
        if not expected_keys:
            result = {str(key): str(value) if value is not None else "" for key, value in parsed.items()}
            result[filename_key] = filename
            return result
        if filename_key not in expected_keys:
            expected_keys = [filename_key, *expected_keys]
        normalized_parsed = {_normalize_key(key): key for key in parsed.keys()}
        result: dict[str, str] = {}
        for expected_key in expected_keys:
            normalized_expected = _normalize_key(expected_key)
            if "filename" in normalized_expected and filename:
                result[expected_key] = filename
                continue
            source_key = normalized_parsed.get(normalized_expected)
            if source_key is not None:
                value = parsed.get(source_key)
                result[expected_key] = str(value) if value is not None else ""
            else:
                result[expected_key] = ""
        return result

    def _request_plain_text(self, settings: dict[str, str], content: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {settings['api_key']}",
            "User-Agent": "RecordPrep/0.1",
        }
        body = {
            "model": settings["model_id"],
            "stream": False,
            "messages": [
                {"role": "system", "content": settings["prompt"]},
                {"role": "user", "content": content},
            ],
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(settings["api_url"], data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
        except urllib.error.HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                error_body = ""
            detail = error_body.strip() or exc.reason or "request failed"
            raise RuntimeError(f"Classifier request failed: HTTP {exc.code} {detail}") from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Classifier request failed: {exc}") from exc
        return self._extract_response_text(payload).strip()

    def _extract_response_text(self, payload: Any) -> str:
        if isinstance(payload, dict):
            choices = payload.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0] or {}
                message = first.get("message") or {}
                content = message.get("content")
                if isinstance(content, str):
                    return content
            for key in ("output", "text", "data"):
                value = payload.get(key)
                if isinstance(value, str):
                    return value
        return ""

    def _extract_json_payload(self, text: str) -> str:
        if not text:
            return ""
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match:
            return match.group(0).strip()
        return stripped


class RecordPrepApp(Adw.Application):
    def __init__(self) -> None:
        super().__init__(application_id=APPLICATION_ID)

    def do_activate(self) -> None:
        win = self.props.active_window
        if not win:
            win = RecordPrepWindow(self)
        win.present()


def main() -> None:
    app = RecordPrepApp()
    app.run(None)


if __name__ == "__main__":
    main()
