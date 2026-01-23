#!/usr/bin/env python3
from __future__ import annotations

import datetime
import os
import importlib
import json
import random
import re
import threading
import time
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

LLM_MAX_RETRIES = 5
LLM_RETRY_BASE_SECONDS = 1.0
LLM_RETRY_MAX_SECONDS = 30.0
LLM_RETRYABLE_HTTP_CODES = {408, 409, 429, 500, 502, 503, 504}


class StopRequested(RuntimeError):
    pass

CONFIG_FILE = Path(__file__).with_name("config.json")
CONFIG_KEY_CLASSIFIER_API_URL = "classifier_api_url"
CONFIG_KEY_CLASSIFIER_MODEL_ID = "classifier_model_id"
CONFIG_KEY_CLASSIFIER_API_KEY = "classifier_api_key"
CONFIG_KEY_CLASSIFIER_PROMPT = "classifier_prompt"
CONFIG_KEY_CLASSIFY_FURTHER_API_URL = "classify_further_api_url"
CONFIG_KEY_CLASSIFY_FURTHER_MODEL_ID = "classify_further_model_id"
CONFIG_KEY_CLASSIFY_FURTHER_API_KEY = "classify_further_api_key"
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
CONFIG_KEY_FORM_FILTER_API_URL = "form_filter_api_url"
CONFIG_KEY_FORM_FILTER_MODEL_ID = "form_filter_model_id"
CONFIG_KEY_FORM_FILTER_API_KEY = "form_filter_api_key"
CONFIG_KEY_FORM_FILTER_PROMPT = "form_filter_prompt"
CONFIG_KEY_REPORT_FILTER_PROMPT = "report_filter_prompt"
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
CONFIG_KEY_SUMMARIZE_MINUTES_PROMPT = "summarize_minutes_prompt"
CONFIG_KEY_SUMMARIZE_CHUNK_SIZE = "summarize_chunk_size"
CONFIG_KEY_OVERVIEW_API_URL = "overview_api_url"
CONFIG_KEY_OVERVIEW_MODEL_ID = "overview_model_id"
CONFIG_KEY_OVERVIEW_API_KEY = "overview_api_key"
CONFIG_KEY_OVERVIEW_PROMPT = "overview_prompt"
CONFIG_KEY_RAG_VOYAGE_API_KEY = "rag_voyage_api_key"
CONFIG_KEY_RAG_VOYAGE_MODEL = "rag_voyage_model"
CONFIG_KEY_SELECTED_PDFS = "selected_pdfs"
DEFAULT_CLASSIFIER_PROMPT = (
    "You are labeling a single page of an OCR'd legal transcript. "
    "Return JSON with keys: page_type, form_name, date. "
    "page_type must be one of: hearing, hearing_first_page, report, form, cover, index, "
    "notice, minute_order, minute_order_first_page, form_first_page. "
    "date should be a long-form U.S. date if present. "
    "form_name should be the form name if page_type is form or form_first_page. "
    "If unknown, use an empty string."
)
DEFAULT_CLASSIFY_DATES_PROMPT = (
    "You are extracting the date from a hearing_first_page or minute_order_first_page page "
    "of an OCR'd legal transcript. "
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
DEFAULT_FORM_FILTER_PROMPT = (
    "You are reviewing a page labeled form_first_page in an OCR'd legal transcript. "
    "Return JSON with keys: is_relevant. "
    "is_relevant must be yes if and only if the page is one of the five specific forms we care about; "
    "otherwise return no."
)
DEFAULT_REPORT_FILTER_PROMPT = (
    "You are reviewing a page labeled report in an OCR'd legal transcript. "
    "Return JSON with keys: is_relevant. "
    "is_relevant must be yes if and only if the report is one of the relevant report titles; "
    "otherwise return no."
)
DEFAULT_CLASSIFY_FORM_NAMES_PROMPT = (
    "You are extracting the form name from a form_first_page in an OCR'd legal transcript. "
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
DEFAULT_SUMMARIZE_MINUTES_PROMPT = (
    "I will provide you with the pages of a minute order. Based on this information, "
    "state the name of the hearing, whether the hearing was reported, whether one or "
    "both parents were present, and what the juvenile court ordered. The description "
    "of what the juvenile court ordered must be brief and concise. Only state that a "
    "parent is present if the minute order indicates that the parent is present on the "
    "first page of the minute order. If only a parent's attorney is listed, assume that "
    "the parent is not present. Do not insert any line breaks. Here are three examples "
    "of the proper format:\n\nDetention Hearing. Reported. No parent appeared. The "
    "juvenile court ordered the children temporarily removed from the parents.\n\n"
    "Receipt of Report Hearing. Not Reported. No parent appeared. The juvenile court "
    "received the section 361.66 report into evidence.\n\nPermanent Plan Review "
    "Hearing. Reported. Only mother appeared. The juvenile court received the social "
    "worker reports into evidence and heard testimony from mother. The juvenile court "
    "terminated parental rights.\n\nOkay, here is the minute order:"
)
DEFAULT_SUMMARIZE_CHUNK_SIZE = 15
DEFAULT_OVERVIEW_PROMPT = (
    "I will provide you with summaries from a legal case. Please provide concise "
    "details about the case in the form of three paragraphs. In the first paragraph, "
    "identify the parties and specify which attorney represented them. Identify each "
    "attorney by name rather than just their law firm. In the second paragraph, "
    "provide a procedural history of the case. In the third paragraph, provide a "
    "factual history of the case. Do not add any other commentary. Okay, here are the "
    "summaries:"
)
DEFAULT_RAG_VOYAGE_MODEL = "voyage-law-2"


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


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"true", "yes", "y", "1", "relevant", "keep"}


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
    for file_name, page_type, page_number in entries:
        if page_type in {"hearing_first_page", "minute_order_first_page"}:
            targets.append((file_name, page_type))
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
            if not file_name or page_type != "form_first_page":
                continue
            targets.append((file_name, page_type))
    return targets


def _load_relevant_form_targets(path: Path) -> list[str]:
    entries = _load_jsonl_entries(path)
    targets: list[str] = []
    for entry in entries:
        file_name = _extract_entry_value(entry, "file_name", "filename")
        if file_name:
            targets.append(file_name)
    return targets


def _load_relevant_report_targets(path: Path) -> list[str]:
    entries = _load_jsonl_entries(path)
    targets: list[str] = []
    for entry in entries:
        file_name = _extract_entry_value(entry, "file_name", "filename")
        if file_name:
            targets.append(file_name)
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


def _load_jsonl_file_names(path: Path) -> set[str]:
    entries = _load_jsonl_entries(path)
    file_names: set[str] = set()
    for entry in entries:
        file_name = _extract_entry_value(entry, "file_name", "filename")
        if file_name:
            file_names.add(file_name)
    return file_names


def _load_indexed_jsonl(path: Path) -> dict[int, str]:
    entries: dict[int, str] = {}
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
            if not isinstance(payload, dict):
                continue
            index_value = payload.get("index")
            try:
                index = int(index_value)
            except (TypeError, ValueError):
                continue
            text = payload.get("text")
            entries[index] = str(text) if text is not None else ""
    return entries


def _append_indexed_jsonl(path: Path, index: int, text: str, label: str | None = None) -> None:
    payload: dict[str, Any] = {"index": index, "text": text}
    if label:
        payload["label"] = label
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


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


def _load_case_name_from_file(root_dir: Path) -> str:
    case_name_path = root_dir / "case_name.txt"
    if not case_name_path.exists():
        return ""
    try:
        value = case_name_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    return _sanitize_case_name_value(value)


def _summary_output_paths(root_dir: Path) -> tuple[Path, Path]:
    summaries_dir = root_dir / "summaries"
    case_name = _load_case_name_from_file(root_dir)
    if not case_name:
        case_name, _ = load_case_context()
        case_name = _sanitize_case_name_value(case_name)
    if case_name:
        return (
            summaries_dir / f"hearings_sum_{case_name}.txt",
            summaries_dir / f"reports_sum_{case_name}.txt",
        )
    return (
        summaries_dir / "summarized_hearings.txt",
        summaries_dir / "summarized_reports.txt",
    )


def _minutes_summary_output_path(root_dir: Path) -> Path:
    summaries_dir = root_dir / "summaries"
    case_name = _load_case_name_from_file(root_dir)
    if not case_name:
        case_name, _ = load_case_context()
        case_name = _sanitize_case_name_value(case_name)
    if case_name:
        return summaries_dir / f"minutes_sum_{case_name}.txt"
    return summaries_dir / "summarized_minutes.txt"


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


def _ensure_case_bundle_dirs(base_dir: Path) -> tuple[Path, Path, Path]:
    root = base_dir / "case_bundle"
    text_dir = root / "text_pages"
    image_pages_dir = root / "image_pages"
    text_dir.mkdir(parents=True, exist_ok=True)
    image_pages_dir.mkdir(parents=True, exist_ok=True)
    return root, text_dir, image_pages_dir


def _checkpoint_dir(root_dir: Path) -> Path:
    path = root_dir / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _retry_after_seconds(error: urllib.error.HTTPError) -> float | None:
    if not error.headers:
        return None
    retry_after = error.headers.get("Retry-After")
    if not retry_after:
        return None
    try:
        return float(retry_after)
    except ValueError:
        return None


def _retry_delay_seconds(attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return max(0.0, retry_after)
    base = min(LLM_RETRY_MAX_SECONDS, LLM_RETRY_BASE_SECONDS * (2 ** (attempt - 1)))
    jitter = random.uniform(0.0, base * 0.2)
    return base + jitter


def _post_json_with_retries(
    req: urllib.request.Request,
    timeout: int,
    error_label: str,
) -> dict[str, Any]:
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
            if isinstance(payload, dict):
                return payload
            raise RuntimeError(f"{error_label}: response was not JSON")
        except urllib.error.HTTPError as exc:
            retry_after = _retry_after_seconds(exc)
            if exc.code in LLM_RETRYABLE_HTTP_CODES and attempt < LLM_MAX_RETRIES:
                time.sleep(_retry_delay_seconds(attempt, retry_after))
                continue
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                error_body = ""
            detail = error_body.strip() or exc.reason or "request failed"
            raise RuntimeError(f"{error_label}: HTTP {exc.code} {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < LLM_MAX_RETRIES:
                time.sleep(_retry_delay_seconds(attempt, None))
                continue
            raise RuntimeError(f"{error_label}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"{error_label}: {exc}") from exc
    raise RuntimeError(f"{error_label}: exhausted retries")


def _manifest_path(root_dir: Path) -> Path:
    return root_dir / "manifest.json"


def _read_manifest(root_dir: Path) -> dict[str, Any]:
    manifest_path = _manifest_path(root_dir)
    if not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_manifest(
    root_dir: Path,
    selected_pdfs: list[Path],
    pipeline_info: dict[str, Any] | None = None,
) -> None:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest_path = _manifest_path(root_dir)
    existing = _read_manifest(root_dir)
    created_at = now
    if isinstance(existing, dict):
        existing_created = existing.get("created_at")
        if isinstance(existing_created, str) and existing_created.strip():
            created_at = existing_created

    text_dir = root_dir / "text_pages"
    image_pages_dir = root_dir / "image_pages"
    classification_dir = root_dir / "classification"
    artifacts_dir = root_dir / "artifacts"
    summaries_dir = root_dir / "summaries"
    rag_dir = root_dir / "rag"
    temp_dir = root_dir / "temp"
    summarized_hearings_path, summarized_reports_path = _summary_output_paths(root_dir)
    summarized_minutes_path = _minutes_summary_output_path(root_dir)

    def _root_path(value: Path) -> str:
        return str(value)

    def _relpath(value: Path) -> str:
        if not value.is_absolute():
            return value.as_posix()
        try:
            return value.relative_to(root_dir).as_posix()
        except ValueError:
            return os.path.relpath(str(value), str(root_dir))

    pipeline: dict[str, Any] = {}
    if isinstance(existing.get("pipeline"), dict):
        pipeline.update(existing["pipeline"])
    if pipeline_info:
        for key, value in pipeline_info.items():
            if value is None:
                pipeline.pop(key, None)
            else:
                pipeline[key] = value
        if "last_completed_step" in pipeline_info and "last_completed_at" not in pipeline_info:
            pipeline["last_completed_at"] = now
        if "last_failed_step" in pipeline_info and "last_failed_at" not in pipeline_info:
            pipeline["last_failed_at"] = now

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": created_at,
        "updated_at": now,
        "root_dir": _root_path(root_dir),
        "input_pdfs": [_relpath(path) for path in selected_pdfs],
        "dirs": {
            "text_pages": _relpath(text_dir),
            "image_pages": _relpath(image_pages_dir),
            "classification": _relpath(classification_dir),
            "artifacts": _relpath(artifacts_dir),
            "summaries": _relpath(summaries_dir),
            "rag": _relpath(rag_dir),
            "temp": _relpath(temp_dir),
        },
        "files": {
            "merged_pdf": _relpath(temp_dir / "merged.pdf"),
            "toc": _relpath(artifacts_dir / "toc.txt"),
            "hearing_boundaries": _relpath(artifacts_dir / "hearing_boundaries.json"),
            "report_boundaries": _relpath(artifacts_dir / "report_boundaries.json"),
            "minutes_boundaries": _relpath(artifacts_dir / "minutes_boundaries.json"),
            "raw_hearings": _relpath(artifacts_dir / "raw_hearings.txt"),
            "raw_reports": _relpath(artifacts_dir / "raw_reports.txt"),
            "optimized_hearings": _relpath(artifacts_dir / "optimized_hearings.txt"),
            "optimized_reports": _relpath(artifacts_dir / "optimized_reports.txt"),
            "summarized_hearings": _relpath(summarized_hearings_path),
            "summarized_reports": _relpath(summarized_reports_path),
            "summarized_minutes": _relpath(summarized_minutes_path),
            "case_overview": _relpath(rag_dir / "case_overview.txt"),
        },
        "classification": {
            "basic": _relpath(classification_dir / "basic.jsonl"),
            "basic_corrected": _relpath(classification_dir / "basic_corrected.jsonl"),
            "dates": _relpath(classification_dir / "dates.jsonl"),
            "report_names": _relpath(classification_dir / "report_names.jsonl"),
            "relevant_forms": _relpath(classification_dir / "relevant_forms.jsonl"),
            "relevant_reports": _relpath(classification_dir / "relevant_reports.jsonl"),
            "form_names": _relpath(classification_dir / "form_names.jsonl"),
        },
        "rag": {
            "vector_database": _relpath(rag_dir / "vector_database"),
        },
        "pipeline": pipeline,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def load_classify_further_settings() -> dict[str, str]:
    config = _read_config()
    api_url = str(config.get(CONFIG_KEY_CLASSIFY_FURTHER_API_URL, "") or "").strip()
    model_id = str(config.get(CONFIG_KEY_CLASSIFY_FURTHER_MODEL_ID, "") or "").strip()
    api_key = str(config.get(CONFIG_KEY_CLASSIFY_FURTHER_API_KEY, "") or "").strip()
    dates_prompt = str(
        config.get(CONFIG_KEY_CLASSIFY_DATES_PROMPT, DEFAULT_CLASSIFY_DATES_PROMPT) or ""
    ).strip()
    reports_prompt = str(
        config.get(CONFIG_KEY_CLASSIFY_REPORTS_PROMPT, DEFAULT_CLASSIFY_REPORT_NAMES_PROMPT) or ""
    ).strip()
    forms_prompt = str(
        config.get(CONFIG_KEY_CLASSIFY_FORMS_PROMPT, DEFAULT_CLASSIFY_FORM_NAMES_PROMPT) or ""
    ).strip()
    if not (api_url and model_id and api_key):
        shared = load_classifier_settings()
        api_url = api_url or shared["api_url"]
        model_id = model_id or shared["model_id"]
        api_key = api_key or shared["api_key"]
    return {
        "api_url": api_url,
        "model_id": model_id,
        "api_key": api_key,
        "dates_prompt": dates_prompt or DEFAULT_CLASSIFY_DATES_PROMPT,
        "reports_prompt": reports_prompt or DEFAULT_CLASSIFY_REPORT_NAMES_PROMPT,
        "forms_prompt": forms_prompt or DEFAULT_CLASSIFY_FORM_NAMES_PROMPT,
    }


def load_form_filter_settings() -> dict[str, str]:
    config = _read_config()
    api_url = str(config.get(CONFIG_KEY_FORM_FILTER_API_URL, "") or "").strip()
    model_id = str(config.get(CONFIG_KEY_FORM_FILTER_MODEL_ID, "") or "").strip()
    api_key = str(config.get(CONFIG_KEY_FORM_FILTER_API_KEY, "") or "").strip()
    prompt = str(config.get(CONFIG_KEY_FORM_FILTER_PROMPT, DEFAULT_FORM_FILTER_PROMPT) or "").strip()
    report_prompt = str(
        config.get(CONFIG_KEY_REPORT_FILTER_PROMPT, DEFAULT_REPORT_FILTER_PROMPT) or ""
    ).strip()
    return {
        "api_url": api_url,
        "model_id": model_id,
        "api_key": api_key,
        "prompt": prompt or DEFAULT_FORM_FILTER_PROMPT,
        "report_prompt": report_prompt or DEFAULT_REPORT_FILTER_PROMPT,
    }


def save_form_filter_settings(
    api_url: str,
    model_id: str,
    api_key: str,
    prompt: str,
    report_prompt: str,
) -> None:
    config = _read_config()
    config[CONFIG_KEY_FORM_FILTER_API_URL] = api_url
    config[CONFIG_KEY_FORM_FILTER_MODEL_ID] = model_id
    config[CONFIG_KEY_FORM_FILTER_API_KEY] = api_key
    config[CONFIG_KEY_FORM_FILTER_PROMPT] = prompt or DEFAULT_FORM_FILTER_PROMPT
    config[CONFIG_KEY_REPORT_FILTER_PROMPT] = report_prompt or DEFAULT_REPORT_FILTER_PROMPT
    _write_config(config)


def save_classify_further_settings(api_url: str, model_id: str, api_key: str) -> None:
    config = _read_config()
    config[CONFIG_KEY_CLASSIFY_FURTHER_API_URL] = api_url
    config[CONFIG_KEY_CLASSIFY_FURTHER_MODEL_ID] = model_id
    config[CONFIG_KEY_CLASSIFY_FURTHER_API_KEY] = api_key
    _write_config(config)


def save_classify_further_prompts(
    dates_prompt: str,
    reports_prompt: str,
    forms_prompt: str,
) -> None:
    config = _read_config()
    config[CONFIG_KEY_CLASSIFY_DATES_PROMPT] = dates_prompt or DEFAULT_CLASSIFY_DATES_PROMPT
    config[CONFIG_KEY_CLASSIFY_REPORTS_PROMPT] = (
        reports_prompt or DEFAULT_CLASSIFY_REPORT_NAMES_PROMPT
    )
    config[CONFIG_KEY_CLASSIFY_FORMS_PROMPT] = forms_prompt or DEFAULT_CLASSIFY_FORM_NAMES_PROMPT
    _write_config(config)


def load_classify_dates_settings() -> dict[str, str]:
    config = _read_config()
    api_url = str(config.get(CONFIG_KEY_CLASSIFY_DATES_API_URL, "") or "").strip()
    model_id = str(config.get(CONFIG_KEY_CLASSIFY_DATES_MODEL_ID, "") or "").strip()
    api_key = str(config.get(CONFIG_KEY_CLASSIFY_DATES_API_KEY, "") or "").strip()
    prompt = str(config.get(CONFIG_KEY_CLASSIFY_DATES_PROMPT, DEFAULT_CLASSIFY_DATES_PROMPT) or "").strip()
    if not (api_url and model_id and api_key):
        shared = load_classify_further_settings()
        api_url = api_url or shared["api_url"]
        model_id = model_id or shared["model_id"]
        api_key = api_key or shared["api_key"]
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
    if not (api_url and model_id and api_key):
        shared = load_classify_further_settings()
        api_url = api_url or shared["api_url"]
        model_id = model_id or shared["model_id"]
        api_key = api_key or shared["api_key"]
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
    if not (api_url and model_id and api_key):
        shared = load_classify_further_settings()
        api_url = api_url or shared["api_url"]
        model_id = model_id or shared["model_id"]
        api_key = api_key or shared["api_key"]
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


def _generate_image_page_files(pdf_path: Path, image_pages_dir: Path) -> None:
    doc = fitz.open(str(pdf_path))
    try:
        for index in range(len(doc)):
            page = doc.load_page(index)
            pix = page.get_pixmap(dpi=300, colorspace=fitz.csGRAY)
            pix.save(str(image_pages_dir / f"{index + 1:04d}.png"))
    finally:
        doc.close()

@dataclass
class ClassifySettingsWidgets:
    api_url_row: Adw.EntryRow
    model_row: Adw.EntryRow
    api_key_row: Adw.EntryRow
    prompt_buffer: Gtk.TextBuffer


@dataclass
class ClassifyFurtherSettingsWidgets:
    api_url_row: Adw.EntryRow
    model_row: Adw.EntryRow
    api_key_row: Adw.EntryRow
    dates_prompt_buffer: Gtk.TextBuffer
    reports_prompt_buffer: Gtk.TextBuffer
    forms_prompt_buffer: Gtk.TextBuffer


@dataclass
class ClassifyFilterSettingsWidgets:
    api_url_row: Adw.EntryRow
    model_row: Adw.EntryRow
    api_key_row: Adw.EntryRow
    form_prompt_buffer: Gtk.TextBuffer
    report_prompt_buffer: Gtk.TextBuffer


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
    minutes_prompt_buffer: Gtk.TextBuffer


@dataclass
class OverviewSettingsWidgets:
    api_url_row: Adw.EntryRow
    model_row: Adw.EntryRow
    api_key_row: Adw.EntryRow
    prompt_buffer: Gtk.TextBuffer


@dataclass
class RagSettingsWidgets:
    voyage_model_row: Adw.EntryRow
    voyage_key_row: Adw.EntryRow


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
    minutes_prompt = str(
        config.get(CONFIG_KEY_SUMMARIZE_MINUTES_PROMPT, DEFAULT_SUMMARIZE_MINUTES_PROMPT) or ""
    ).strip()
    return {
        "api_url": api_url,
        "model_id": model_id,
        "api_key": api_key,
        "chunk_size": str(chunk_size),
        "hearings_prompt": hearings_prompt or DEFAULT_SUMMARIZE_HEARINGS_PROMPT,
        "reports_prompt": reports_prompt or DEFAULT_SUMMARIZE_REPORTS_PROMPT,
        "minutes_prompt": minutes_prompt or DEFAULT_SUMMARIZE_MINUTES_PROMPT,
    }


def save_summarize_settings(
    api_url: str,
    model_id: str,
    api_key: str,
    chunk_size: str,
    hearings_prompt: str,
    reports_prompt: str,
    minutes_prompt: str,
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
    config[CONFIG_KEY_SUMMARIZE_MINUTES_PROMPT] = (
        minutes_prompt or DEFAULT_SUMMARIZE_MINUTES_PROMPT
    )
    _write_config(config)


def load_overview_settings() -> dict[str, str]:
    config = _read_config()
    api_url = str(config.get(CONFIG_KEY_OVERVIEW_API_URL, "") or "").strip()
    model_id = str(config.get(CONFIG_KEY_OVERVIEW_MODEL_ID, "") or "").strip()
    api_key = str(config.get(CONFIG_KEY_OVERVIEW_API_KEY, "") or "").strip()
    prompt = str(config.get(CONFIG_KEY_OVERVIEW_PROMPT, DEFAULT_OVERVIEW_PROMPT) or "").strip()
    return {
        "api_url": api_url,
        "model_id": model_id,
        "api_key": api_key,
        "prompt": prompt or DEFAULT_OVERVIEW_PROMPT,
    }


def save_overview_settings(
    api_url: str,
    model_id: str,
    api_key: str,
    prompt: str,
) -> None:
    config = _read_config()
    config[CONFIG_KEY_OVERVIEW_API_URL] = api_url
    config[CONFIG_KEY_OVERVIEW_MODEL_ID] = model_id
    config[CONFIG_KEY_OVERVIEW_API_KEY] = api_key
    config[CONFIG_KEY_OVERVIEW_PROMPT] = prompt or DEFAULT_OVERVIEW_PROMPT
    _write_config(config)


def load_rag_settings() -> dict[str, str]:
    config = _read_config()
    voyage_key = str(config.get(CONFIG_KEY_RAG_VOYAGE_API_KEY, "") or "").strip()
    voyage_model = str(
        config.get(CONFIG_KEY_RAG_VOYAGE_MODEL, DEFAULT_RAG_VOYAGE_MODEL) or ""
    ).strip()
    return {
        "voyage_api_key": voyage_key,
        "voyage_model": voyage_model or DEFAULT_RAG_VOYAGE_MODEL,
    }


def save_rag_settings(
    voyage_api_key: str,
    voyage_model: str,
) -> None:
    config = _read_config()
    config[CONFIG_KEY_RAG_VOYAGE_API_KEY] = voyage_api_key
    config[CONFIG_KEY_RAG_VOYAGE_MODEL] = voyage_model or DEFAULT_RAG_VOYAGE_MODEL
    _write_config(config)


class SettingsWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application, on_saved: Callable[[], None] | None = None) -> None:
        super().__init__(application=app, title="Settings")
        self.set_default_size(900, 720)
        self.set_resizable(True)
        self._on_saved = on_saved
        self._prompt_editors: dict[str, ClassifySettingsWidgets] = {}
        self._classify_further_widgets: ClassifyFurtherSettingsWidgets | None = None
        self._classify_filter_widgets: ClassifyFilterSettingsWidgets | None = None
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

        classify_row = Gtk.ListBoxRow()
        classify_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        classify_box.set_margin_top(8)
        classify_box.set_margin_bottom(8)
        classify_box.set_margin_start(12)
        classify_box.set_margin_end(12)
        classify_label = Gtk.Label(label="Classification basic", xalign=0)
        classify_box.append(classify_label)
        classify_row.set_child(classify_box)
        prompt_list.append(classify_row)
        self._prompt_row_keys[classify_row] = "classify-basic"
        classify_page = self._build_prompt_page(
            "classify-basic",
            "Classification basic",
            load_classifier_settings(),
            DEFAULT_CLASSIFIER_PROMPT,
        )
        prompt_stack.add_named(classify_page, "classify-basic")

        classify_filter_row = Gtk.ListBoxRow()
        classify_filter_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        classify_filter_box.set_margin_top(8)
        classify_filter_box.set_margin_bottom(8)
        classify_filter_box.set_margin_start(12)
        classify_filter_box.set_margin_end(12)
        classify_filter_label = Gtk.Label(label="Classification filter", xalign=0)
        classify_filter_box.append(classify_filter_label)
        classify_filter_row.set_child(classify_filter_box)
        prompt_list.append(classify_filter_row)
        self._prompt_row_keys[classify_filter_row] = "classify-filter"
        classify_filter_page = self._build_classify_filter_prompt_page()
        prompt_stack.add_named(classify_filter_page, "classify-filter")

        classify_further_row = Gtk.ListBoxRow()
        classify_further_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        classify_further_box.set_margin_top(8)
        classify_further_box.set_margin_bottom(8)
        classify_further_box.set_margin_start(12)
        classify_further_box.set_margin_end(12)
        classify_further_label = Gtk.Label(label="Classification dates/names", xalign=0)
        classify_further_box.append(classify_further_label)
        classify_further_row.set_child(classify_further_box)
        prompt_list.append(classify_further_row)
        self._prompt_row_keys[classify_further_row] = "classify-further"
        classify_further_page = self._build_classify_further_prompt_page()
        prompt_stack.add_named(classify_further_page, "classify-further")

        optimize_row = Gtk.ListBoxRow()
        optimize_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        optimize_box.set_margin_top(8)
        optimize_box.set_margin_bottom(8)
        optimize_box.set_margin_start(12)
        optimize_box.set_margin_end(12)
        optimize_label = Gtk.Label(label="Optimize", xalign=0)
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

        overview_row = Gtk.ListBoxRow()
        overview_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        overview_box.set_margin_top(8)
        overview_box.set_margin_bottom(8)
        overview_box.set_margin_start(12)
        overview_box.set_margin_end(12)
        overview_label = Gtk.Label(label="Case Overview", xalign=0)
        overview_box.append(overview_label)
        overview_row.set_child(overview_box)
        prompt_list.append(overview_row)
        self._prompt_row_keys[overview_row] = "overview"
        overview_page = self._build_overview_prompt_page(load_overview_settings())
        prompt_stack.add_named(overview_page, "overview")

        rag_row = Gtk.ListBoxRow()
        rag_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        rag_box.set_margin_top(8)
        rag_box.set_margin_bottom(8)
        rag_box.set_margin_start(12)
        rag_box.set_margin_end(12)
        rag_label = Gtk.Label(label="RAG", xalign=0)
        rag_box.append(rag_label)
        rag_row.set_child(rag_box)
        prompt_list.append(rag_row)
        self._prompt_row_keys[rag_row] = "rag"
        rag_page = self._build_rag_prompt_page(load_rag_settings())
        prompt_stack.add_named(rag_page, "rag")

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

    def _build_classify_filter_prompt_page(self) -> Gtk.Widget:
        settings = load_form_filter_settings()

        page_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        page_box.set_margin_top(12)
        page_box.set_margin_bottom(12)
        page_box.set_margin_start(12)
        page_box.set_margin_end(12)
        page_box.set_vexpand(True)

        title_label = Gtk.Label(label="Classification filter", xalign=0)
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

        prompt_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        prompt_section.set_hexpand(True)
        prompt_section.set_vexpand(True)

        forms_label = Gtk.Label(label="Form Relevance Prompt", xalign=0)
        forms_label.add_css_class("dim-label")
        prompt_section.append(forms_label)
        forms_scroller, forms_buffer = self._build_prompt_editor(
            settings.get("prompt") or DEFAULT_FORM_FILTER_PROMPT
        )
        prompt_section.append(forms_scroller)

        reports_label = Gtk.Label(label="Report Relevance Prompt", xalign=0)
        reports_label.add_css_class("dim-label")
        prompt_section.append(reports_label)
        reports_scroller, reports_buffer = self._build_prompt_editor(
            settings.get("report_prompt") or DEFAULT_REPORT_FILTER_PROMPT
        )
        prompt_section.append(reports_scroller)

        page_box.append(prompt_section)

        page = Gtk.ScrolledWindow()
        page.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        page.set_hexpand(True)
        page.set_vexpand(True)
        page.set_child(page_box)

        self._classify_filter_widgets = ClassifyFilterSettingsWidgets(
            api_url_row=api_url_row,
            model_row=model_row,
            api_key_row=api_key_row,
            form_prompt_buffer=forms_buffer,
            report_prompt_buffer=reports_buffer,
        )
        return page

    def _build_classify_further_prompt_page(self) -> Gtk.Widget:
        settings = load_classify_further_settings()

        page_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        page_box.set_margin_top(12)
        page_box.set_margin_bottom(12)
        page_box.set_margin_start(12)
        page_box.set_margin_end(12)
        page_box.set_vexpand(True)

        title_label = Gtk.Label(label="Classification dates/names", xalign=0)
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

        prompt_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        prompt_section.set_hexpand(True)
        prompt_section.set_vexpand(True)

        dates_label = Gtk.Label(label="Classify Dates (First Pages) Prompt", xalign=0)
        dates_label.add_css_class("dim-label")
        prompt_section.append(dates_label)
        dates_scroller, dates_buffer = self._build_prompt_editor(
            settings.get("dates_prompt") or DEFAULT_CLASSIFY_DATES_PROMPT
        )
        prompt_section.append(dates_scroller)

        reports_label = Gtk.Label(label="Classify Report Names (First Pages) Prompt", xalign=0)
        reports_label.add_css_class("dim-label")
        prompt_section.append(reports_label)
        reports_scroller, reports_buffer = self._build_prompt_editor(
            settings.get("reports_prompt") or DEFAULT_CLASSIFY_REPORT_NAMES_PROMPT
        )
        prompt_section.append(reports_scroller)

        forms_label = Gtk.Label(label="Classify Form Names (First Pages) Prompt", xalign=0)
        forms_label.add_css_class("dim-label")
        prompt_section.append(forms_label)
        forms_scroller, forms_buffer = self._build_prompt_editor(
            settings.get("forms_prompt") or DEFAULT_CLASSIFY_FORM_NAMES_PROMPT
        )
        prompt_section.append(forms_scroller)

        page_box.append(prompt_section)

        page = Gtk.ScrolledWindow()
        page.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        page.set_hexpand(True)
        page.set_vexpand(True)
        page.set_child(page_box)

        self._classify_further_widgets = ClassifyFurtherSettingsWidgets(
            api_url_row=api_url_row,
            model_row=model_row,
            api_key_row=api_key_row,
            dates_prompt_buffer=dates_buffer,
            reports_prompt_buffer=reports_buffer,
            forms_prompt_buffer=forms_buffer,
        )
        return page

    def _build_optimize_prompt_page(self, settings: dict[str, str]) -> Gtk.Widget:
        page_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        page_box.set_margin_top(12)
        page_box.set_margin_bottom(12)
        page_box.set_margin_start(12)
        page_box.set_margin_end(12)
        page_box.set_vexpand(True)

        title_label = Gtk.Label(label="Optimize", xalign=0)
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

        minutes_label = Gtk.Label(label="Summarize Minute Orders Prompt", xalign=0)
        minutes_label.add_css_class("dim-label")
        prompt_section.append(minutes_label)
        minutes_scroller, minutes_buffer = self._build_prompt_editor(
            settings.get("minutes_prompt") or DEFAULT_SUMMARIZE_MINUTES_PROMPT
        )
        prompt_section.append(minutes_scroller)

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
            minutes_prompt_buffer=minutes_buffer,
        )
        return page

    def _build_overview_prompt_page(self, settings: dict[str, str]) -> Gtk.Widget:
        page_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        page_box.set_margin_top(12)
        page_box.set_margin_bottom(12)
        page_box.set_margin_start(12)
        page_box.set_margin_end(12)
        page_box.set_vexpand(True)

        title_label = Gtk.Label(label="Case Overview", xalign=0)
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

        prompt_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        prompt_section.set_hexpand(True)
        prompt_section.set_vexpand(True)
        prompt_label = Gtk.Label(label="Prompt", xalign=0)
        prompt_label.add_css_class("dim-label")
        prompt_section.append(prompt_label)
        prompt_scroller, buffer = self._build_prompt_editor(
            settings.get("prompt") or DEFAULT_OVERVIEW_PROMPT
        )
        prompt_section.append(prompt_scroller)
        page_box.append(prompt_section)

        page = Gtk.ScrolledWindow()
        page.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        page.set_hexpand(True)
        page.set_vexpand(True)
        page.set_child(page_box)

        self._overview_widgets = OverviewSettingsWidgets(
            api_url_row=api_url_row,
            model_row=model_row,
            api_key_row=api_key_row,
            prompt_buffer=buffer,
        )
        return page

    def _build_rag_prompt_page(self, settings: dict[str, str]) -> Gtk.Widget:
        page_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        page_box.set_margin_top(12)
        page_box.set_margin_bottom(12)
        page_box.set_margin_start(12)
        page_box.set_margin_end(12)
        page_box.set_vexpand(True)

        title_label = Gtk.Label(label="RAG", xalign=0)
        title_label.add_css_class("title-3")
        page_box.append(title_label)

        voyage_group = Adw.PreferencesGroup(title="Voyage RAG")
        voyage_group.add_css_class("list-stack")
        voyage_group.set_hexpand(True)
        page_box.append(voyage_group)

        voyage_model_row = Adw.EntryRow(title="Voyage Model")
        voyage_model_row.set_text(settings.get("voyage_model", DEFAULT_RAG_VOYAGE_MODEL))
        voyage_group.add(voyage_model_row)

        voyage_key_row = self._build_password_row("Voyage API Key")
        voyage_key_row.set_text(settings.get("voyage_api_key", ""))
        voyage_group.add(voyage_key_row)

        page = Gtk.ScrolledWindow()
        page.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        page.set_hexpand(True)
        page.set_vexpand(True)
        page.set_child(page_box)

        self._rag_widgets = RagSettingsWidgets(
            voyage_model_row=voyage_model_row,
            voyage_key_row=voyage_key_row,
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
        classify_basic_widgets = self._prompt_editors.get("classify-basic")
        classify_filter_widgets = self._classify_filter_widgets
        classify_further_widgets = self._classify_further_widgets
        optimize_widgets = getattr(self, "_optimize_widgets", None)
        summarize_widgets = getattr(self, "_summarize_widgets", None)
        overview_widgets = getattr(self, "_overview_widgets", None)
        rag_widgets = getattr(self, "_rag_widgets", None)
        if case_widgets:
            save_case_name_settings(
                case_widgets.api_url_row.get_text().strip(),
                case_widgets.model_row.get_text().strip(),
                case_widgets.api_key_row.get_text().strip(),
                self._prompt_text(case_widgets.prompt_buffer).strip(),
            )
        if classify_basic_widgets:
            save_classifier_settings(
                classify_basic_widgets.api_url_row.get_text().strip(),
                classify_basic_widgets.model_row.get_text().strip(),
                classify_basic_widgets.api_key_row.get_text().strip(),
                self._prompt_text(classify_basic_widgets.prompt_buffer).strip(),
            )
        if classify_filter_widgets:
            save_form_filter_settings(
                classify_filter_widgets.api_url_row.get_text().strip(),
                classify_filter_widgets.model_row.get_text().strip(),
                classify_filter_widgets.api_key_row.get_text().strip(),
                self._prompt_text(classify_filter_widgets.form_prompt_buffer).strip(),
                self._prompt_text(classify_filter_widgets.report_prompt_buffer).strip(),
            )
        if classify_further_widgets:
            api_url = classify_further_widgets.api_url_row.get_text().strip()
            model_id = classify_further_widgets.model_row.get_text().strip()
            api_key = classify_further_widgets.api_key_row.get_text().strip()
            save_classify_further_settings(api_url, model_id, api_key)
            save_classify_further_prompts(
                self._prompt_text(classify_further_widgets.dates_prompt_buffer).strip(),
                self._prompt_text(classify_further_widgets.reports_prompt_buffer).strip(),
                self._prompt_text(classify_further_widgets.forms_prompt_buffer).strip(),
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
                self._prompt_text(summarize_widgets.minutes_prompt_buffer).strip(),
            )
        if overview_widgets:
            save_overview_settings(
                overview_widgets.api_url_row.get_text().strip(),
                overview_widgets.model_row.get_text().strip(),
                overview_widgets.api_key_row.get_text().strip(),
                self._prompt_text(overview_widgets.prompt_buffer).strip(),
            )
        if rag_widgets:
            save_rag_settings(
                rag_widgets.voyage_key_row.get_text().strip(),
                rag_widgets.voyage_model_row.get_text().strip(),
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
        self._pipeline_running = False
        self._stop_event = threading.Event()
        self._step_status_labels: dict[Adw.ActionRow, Gtk.Label] = {}

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

        action_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.run_all_button = Gtk.Button(label="Run all steps")
        self.run_all_button.set_halign(Gtk.Align.START)
        self.run_all_button.connect("clicked", self.on_run_all_clicked)
        action_box.append(self.run_all_button)

        self.stop_button = Gtk.Button(label="Stop")
        self.stop_button.set_halign(Gtk.Align.START)
        self.stop_button.set_sensitive(False)
        self.stop_button.connect("clicked", self.on_stop_clicked)
        action_box.append(self.stop_button)

        self.resume_button = Gtk.Button(label="Resume")
        self.resume_button.set_halign(Gtk.Align.START)
        self.resume_button.connect("clicked", self.on_resume_clicked)
        action_box.append(self.resume_button)
        content.append(action_box)

        self.step_list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        self.step_list.add_css_class("boxed-list")
        content.append(self.step_list)

        self.step_one_row = Adw.ActionRow(
            title="Create files",
            subtitle="Generate per-page text and image files for the selected PDFs.",
        )
        self.step_one_row.set_activatable(True)
        self.step_one_row.connect("activated", self.on_step_one_clicked)
        self._attach_step_status(self.step_one_row)
        self.step_list.append(self.step_one_row)

        self.step_strip_nonstandard_row = Adw.ActionRow(
            title="Strip characters",
            subtitle="Remove non-printing characters from the extracted text files.",
        )
        self.step_strip_nonstandard_row.set_activatable(True)
        self.step_strip_nonstandard_row.connect("activated", self.on_step_strip_nonstandard_clicked)
        self._attach_step_status(self.step_strip_nonstandard_row)
        self.step_list.append(self.step_strip_nonstandard_row)

        self.step_infer_case_row = Adw.ActionRow(
            title="Infer case",
            subtitle="Use the first pages to infer the case name and save it.",
        )
        self.step_infer_case_row.set_activatable(True)
        self.step_infer_case_row.connect("activated", self.on_step_infer_case_clicked)
        self._attach_step_status(self.step_infer_case_row)
        self.step_list.append(self.step_infer_case_row)

        self.step_two_row = Adw.ActionRow(
            title="Classification basic",
            subtitle="Create basic classifications for every page.",
        )
        self.step_two_row.set_activatable(True)
        self.step_two_row.connect("activated", self.on_step_two_clicked)
        self._attach_step_status(self.step_two_row)
        self.step_list.append(self.step_two_row)

        self.step_correct_basic_row = Adw.ActionRow(
            title="Correct basic classification",
            subtitle="Fix common classification mistakes in basic.jsonl.",
        )
        self.step_correct_basic_row.set_activatable(True)
        self.step_correct_basic_row.connect("activated", self.on_step_correct_basic_clicked)
        self._attach_step_status(self.step_correct_basic_row)
        self.step_list.append(self.step_correct_basic_row)

        self.step_filter_row = Adw.ActionRow(
            title="Classification filter",
            subtitle="Identify relevant forms from form first pages.",
        )
        self.step_filter_row.set_activatable(True)
        self.step_filter_row.connect("activated", self.on_step_filter_clicked)
        self._attach_step_status(self.step_filter_row)
        self.step_list.append(self.step_filter_row)

        self.step_three_row = Adw.ActionRow(
            title="Classification dates/names",
            subtitle="Classify dates, report names, and form names from first pages.",
        )
        self.step_three_row.set_activatable(True)
        self.step_three_row.connect("activated", self.on_step_three_clicked)
        self._attach_step_status(self.step_three_row)
        self.step_list.append(self.step_three_row)

        self.step_six_row = Adw.ActionRow(
            title="Build TOC",
            subtitle="Compile a table of contents for forms, reports, orders, and hearings.",
        )
        self.step_six_row.set_activatable(True)
        self.step_six_row.connect("activated", self.on_step_six_clicked)
        self._attach_step_status(self.step_six_row)
        self.step_list.append(self.step_six_row)

        self.step_seven_row = Adw.ActionRow(
            title="Find boundaries",
            subtitle="Determine page ranges for hearing, report, and minute order sections.",
        )
        self.step_seven_row.set_activatable(True)
        self.step_seven_row.connect("activated", self.on_step_seven_clicked)
        self._attach_step_status(self.step_seven_row)
        self.step_list.append(self.step_seven_row)

        self.step_eight_row = Adw.ActionRow(
            title="Create raw",
            subtitle="Create raw hearing and report text files for summarization.",
        )
        self.step_eight_row.set_activatable(True)
        self.step_eight_row.connect("activated", self.on_step_eight_clicked)
        self._attach_step_status(self.step_eight_row)
        self.step_list.append(self.step_eight_row)

        self.step_nine_row = Adw.ActionRow(
            title="Create optimized",
            subtitle="Prepare optimized hearing and report text for retrieval.",
        )
        self.step_nine_row.set_activatable(True)
        self.step_nine_row.connect("activated", self.on_step_nine_clicked)
        self._attach_step_status(self.step_nine_row)
        self.step_list.append(self.step_nine_row)

        self.step_ten_row = Adw.ActionRow(
            title="Create summaries",
            subtitle="Summarize hearings, reports, and minute orders into concise paragraphs.",
        )
        self.step_ten_row.set_activatable(True)
        self.step_ten_row.connect("activated", self.on_step_ten_clicked)
        self._attach_step_status(self.step_ten_row)
        self.step_list.append(self.step_ten_row)

        self.step_eleven_row = Adw.ActionRow(
            title="Case overview",
            subtitle="Create a three-paragraph overview for RAG context.",
        )
        self.step_eleven_row.set_activatable(True)
        self.step_eleven_row.connect("activated", self.on_step_eleven_clicked)
        self._attach_step_status(self.step_eleven_row)
        self.step_list.append(self.step_eleven_row)

        self.step_twelve_row = Adw.ActionRow(
            title="Create RAG index",
            subtitle="Build a VoyageAI/Chroma vector store from optimized text.",
        )
        self.step_twelve_row.set_activatable(True)
        self.step_twelve_row.connect("activated", self.on_step_twelve_clicked)
        self._attach_step_status(self.step_twelve_row)
        self.step_list.append(self.step_twelve_row)

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

    def _attach_step_status(self, row: Adw.ActionRow) -> None:
        status_label = Gtk.Label(label="Pending", xalign=1)
        status_label.add_css_class("dim-label")
        row.add_suffix(status_label)
        self._step_status_labels[row] = status_label

    def _set_step_status(self, row: Adw.ActionRow, status: str) -> None:
        label = self._step_status_labels.get(row)
        if label is not None:
            label.set_text(status)

    def _reset_step_statuses(self) -> None:
        for row in self._step_status_labels:
            self._set_step_status(row, "Pending")

    def _finish_step(self, row: Adw.ActionRow, success: bool | None) -> None:
        if success is None:
            self._set_step_status(row, "Stopped")
        else:
            self._set_step_status(row, "Done" if success else "Failed")

    def _start_step(self, row: Adw.ActionRow) -> None:
        title = row.get_title() or "Working"
        self._set_step_status(row, "Running")
        self._set_status(f"Working: {title}", True)

    def _stop_status(self) -> None:
        self._set_status(APPLICATION_NAME, False)

    def _stop_status_if_idle(self) -> None:
        if not self._pipeline_running:
            self._stop_status()

    def _stop_button_if_idle(self) -> None:
        if not self._pipeline_running:
            self.stop_button.set_sensitive(False)

    def _raise_if_stop_requested(self) -> None:
        if self._stop_event.is_set():
            raise StopRequested()

    def on_stop_clicked(self, _button: Gtk.Button) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self.stop_button.set_sensitive(False)
        self.show_toast("Stop requested.")

    def _safe_update_manifest(
        self, root_dir: Path, pipeline_info: dict[str, Any] | None = None
    ) -> None:
        try:
            _write_manifest(root_dir, self.selected_pdfs, pipeline_info=pipeline_info)
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Manifest update failed: {exc}")

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
        self._reset_step_statuses()
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
        self._reset_step_statuses()
        self.selected_label.set_text(f"Selected: {label}")

    def _load_case_context(self) -> None:
        case_name, _root_dir = load_case_context()
        if case_name:
            display_name = case_name.replace("_", " ") if case_name else case_name
            self.selected_label.set_text(f"Selected: {display_name}")

    def _pipeline_steps(self) -> list[tuple[str, Adw.ActionRow, Callable[[], bool]]]:
        return [
            ("create_files", self.step_one_row, self._run_step_one),
            ("strip_characters", self.step_strip_nonstandard_row, self._run_step_strip_nonstandard),
            ("infer_case", self.step_infer_case_row, self._run_step_infer_case),
            ("classify_basic", self.step_two_row, self._run_step_two),
            ("correct_basic", self.step_correct_basic_row, self._run_step_correct_basic),
            ("classify_filter", self.step_filter_row, self._run_step_filter),
            ("classify_further", self.step_three_row, self._run_step_three),
            ("build_toc", self.step_six_row, self._run_step_six),
            ("find_boundaries", self.step_seven_row, self._run_step_seven),
            ("create_raw", self.step_eight_row, self._run_step_eight),
            ("create_optimized", self.step_nine_row, self._run_step_nine),
            ("create_summaries", self.step_ten_row, self._run_step_ten),
            ("case_overview", self.step_eleven_row, self._run_step_eleven),
            ("create_rag_index", self.step_twelve_row, self._run_step_twelve),
        ]

    def _resolve_case_root(self) -> Path | None:
        parents = {path.parent for path in self.selected_pdfs}
        if len(parents) != 1:
            return None
        base_dir = parents.pop()
        return base_dir / "case_bundle"

    def _resume_start_index(
        self,
        steps: list[tuple[str, Adw.ActionRow, Callable[[], bool]]],
        root_dir: Path,
    ) -> int:
        manifest = _read_manifest(root_dir)
        pipeline = manifest.get("pipeline") if isinstance(manifest, dict) else None
        pipeline = pipeline if isinstance(pipeline, dict) else {}
        last_failed = pipeline.get("last_failed_step")
        last_completed = pipeline.get("last_completed_step")
        step_ids = [step_id for step_id, _row, _handler in steps]
        if last_failed == "classify":
            last_failed = "classify_basic"
        if last_completed == "classify":
            last_completed = "classify_further"
        if isinstance(last_failed, str) and last_failed in step_ids:
            return step_ids.index(last_failed)
        if isinstance(last_completed, str) and last_completed in step_ids:
            return step_ids.index(last_completed) + 1
        return 0

    def on_run_all_clicked(self, _button: Gtk.Button) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        if self._pipeline_running:
            self.show_toast("Pipeline already running.")
            return
        self._stop_event.clear()
        self._pipeline_running = True
        self.run_all_button.set_sensitive(False)
        self.stop_button.set_sensitive(True)
        self.resume_button.set_sensitive(False)
        self.step_list.set_sensitive(False)
        threading.Thread(target=self._run_all_steps, daemon=True).start()

    def on_resume_clicked(self, _button: Gtk.Button) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        if self._pipeline_running:
            self.show_toast("Pipeline already running.")
            return
        root_dir = self._resolve_case_root()
        if root_dir is None:
            self.show_toast("Selected PDFs must be in the same folder.")
            return
        steps = self._pipeline_steps()
        start_index = self._resume_start_index(steps, root_dir)
        if start_index >= len(steps):
            self.show_toast("All steps already complete.")
            return
        for _step_id, row, _handler in steps[:start_index]:
            self._set_step_status(row, "Done")
        self._stop_event.clear()
        self._pipeline_running = True
        self.run_all_button.set_sensitive(False)
        self.stop_button.set_sensitive(True)
        self.resume_button.set_sensitive(False)
        self.step_list.set_sensitive(False)
        threading.Thread(
            target=self._run_steps_from_index,
            args=(start_index, root_dir),
            daemon=True,
        ).start()

    def on_step_one_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self._stop_event.clear()
        self.stop_button.set_sensitive(True)
        self.step_one_row.set_sensitive(False)
        self._start_step(self.step_one_row)
        threading.Thread(target=self._run_step_one, daemon=True).start()

    def on_step_strip_nonstandard_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self._stop_event.clear()
        self.stop_button.set_sensitive(True)
        self.step_strip_nonstandard_row.set_sensitive(False)
        self._start_step(self.step_strip_nonstandard_row)
        threading.Thread(target=self._run_step_strip_nonstandard, daemon=True).start()

    def on_step_infer_case_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self._stop_event.clear()
        self.stop_button.set_sensitive(True)
        self.step_infer_case_row.set_sensitive(False)
        self._start_step(self.step_infer_case_row)
        threading.Thread(target=self._run_step_infer_case, daemon=True).start()

    def _run_step_one(self) -> bool:
        success: bool | None = False
        try:
            self._raise_if_stop_requested()
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir, text_dir, image_pages_dir = _ensure_case_bundle_dirs(base_dir)
            if len(self.selected_pdfs) > 1:
                temp_dir = root_dir / "temp"
                temp_dir.mkdir(parents=True, exist_ok=True)
                merged_path = temp_dir / "merged.pdf"
                pdf_path = _merge_pdfs(self.selected_pdfs, merged_path)
            else:
                pdf_path = self.selected_pdfs[0]
            self._raise_if_stop_requested()
            _generate_text_files(pdf_path, text_dir)
            self._raise_if_stop_requested()
            _generate_image_page_files(pdf_path, image_pages_dir)
        except StopRequested:
            success = None
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Create files failed: {exc}")
        else:
            success = True
            self._safe_update_manifest(
                root_dir,
                {
                    "last_completed_step": "create_files",
                    "last_failed_step": None,
                    "last_failed_at": None,
                },
            )
            GLib.idle_add(self.show_toast, "Create files complete.")
        finally:
            GLib.idle_add(self.step_one_row.set_sensitive, True)
            GLib.idle_add(self._finish_step, self.step_one_row, success)
            GLib.idle_add(self._stop_status_if_idle)
            GLib.idle_add(self._stop_button_if_idle)
        return success is True

    def _run_step_strip_nonstandard(self) -> bool:
        success: bool | None = False
        try:
            self._raise_if_stop_requested()
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "case_bundle"
            text_dir = root_dir / "text_pages"
            if not text_dir.exists():
                raise FileNotFoundError("Run Create files to generate text files first.")
            text_files = sorted(text_dir.glob("*.txt"), key=_natural_sort_key)
            if not text_files:
                raise FileNotFoundError("No text files found to sanitize.")
            for text_path in text_files:
                self._raise_if_stop_requested()
                content = text_path.read_text(encoding="utf-8", errors="ignore")
                cleaned = _strip_nonstandard_characters(content)
                if cleaned != content:
                    text_path.write_text(cleaned, encoding="utf-8")
        except StopRequested:
            success = None
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Strip non-standard characters failed: {exc}")
        else:
            success = True
            self._safe_update_manifest(
                root_dir,
                {
                    "last_completed_step": "strip_characters",
                    "last_failed_step": None,
                    "last_failed_at": None,
                },
            )
            GLib.idle_add(self.show_toast, "Strip non-standard characters complete.")
        finally:
            GLib.idle_add(self.step_strip_nonstandard_row.set_sensitive, True)
            GLib.idle_add(self._finish_step, self.step_strip_nonstandard_row, success)
            GLib.idle_add(self._stop_status_if_idle)
            GLib.idle_add(self._stop_button_if_idle)
        return success is True

    def _run_step_infer_case(self) -> bool:
        success: bool | None = False
        try:
            self._raise_if_stop_requested()
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "case_bundle"
            text_dir = root_dir / "text_pages"
            if not text_dir.exists():
                raise FileNotFoundError("Run Create files to generate text files first.")
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
            display_name = case_name.replace("_", " ") if case_name else case_name
            GLib.idle_add(self.selected_label.set_text, f"Selected: {display_name}")
        except StopRequested:
            success = None
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Infer case name failed: {exc}")
        else:
            success = True
            self._safe_update_manifest(
                root_dir,
                {
                    "last_completed_step": "infer_case",
                    "last_failed_step": None,
                    "last_failed_at": None,
                },
            )
            GLib.idle_add(self.show_toast, "Infer case name complete.")
        finally:
            GLib.idle_add(self.step_infer_case_row.set_sensitive, True)
            GLib.idle_add(self._finish_step, self.step_infer_case_row, success)
            GLib.idle_add(self._stop_status_if_idle)
            GLib.idle_add(self._stop_button_if_idle)
        return success is True

    def on_step_two_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self._stop_event.clear()
        self.stop_button.set_sensitive(True)
        self.step_two_row.set_sensitive(False)
        self._start_step(self.step_two_row)
        threading.Thread(target=self._run_step_two, daemon=True).start()

    def on_step_filter_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self._stop_event.clear()
        self.stop_button.set_sensitive(True)
        self.step_filter_row.set_sensitive(False)
        self._start_step(self.step_filter_row)
        threading.Thread(target=self._run_step_filter, daemon=True).start()

    def on_step_correct_basic_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self._stop_event.clear()
        self.stop_button.set_sensitive(True)
        self.step_correct_basic_row.set_sensitive(False)
        self._start_step(self.step_correct_basic_row)
        threading.Thread(target=self._run_step_correct_basic, daemon=True).start()

    def on_step_three_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self._stop_event.clear()
        self.stop_button.set_sensitive(True)
        self.step_three_row.set_sensitive(False)
        self._start_step(self.step_three_row)
        threading.Thread(target=self._run_step_three, daemon=True).start()

    def on_step_six_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self._stop_event.clear()
        self.stop_button.set_sensitive(True)
        self.step_six_row.set_sensitive(False)
        self._start_step(self.step_six_row)
        threading.Thread(target=self._run_step_six, daemon=True).start()

    def on_step_seven_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self._stop_event.clear()
        self.stop_button.set_sensitive(True)
        self.step_seven_row.set_sensitive(False)
        self._start_step(self.step_seven_row)
        threading.Thread(target=self._run_step_seven, daemon=True).start()

    def on_step_eight_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self._stop_event.clear()
        self.stop_button.set_sensitive(True)
        self.step_eight_row.set_sensitive(False)
        self._start_step(self.step_eight_row)
        threading.Thread(target=self._run_step_eight, daemon=True).start()

    def on_step_nine_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self._stop_event.clear()
        self.stop_button.set_sensitive(True)
        self.step_nine_row.set_sensitive(False)
        self._start_step(self.step_nine_row)
        threading.Thread(target=self._run_step_nine, daemon=True).start()

    def on_step_ten_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self._stop_event.clear()
        self.stop_button.set_sensitive(True)
        self.step_ten_row.set_sensitive(False)
        self._start_step(self.step_ten_row)
        threading.Thread(target=self._run_step_ten, daemon=True).start()

    def on_step_eleven_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self._stop_event.clear()
        self.stop_button.set_sensitive(True)
        self.step_eleven_row.set_sensitive(False)
        self._start_step(self.step_eleven_row)
        threading.Thread(target=self._run_step_eleven, daemon=True).start()

    def on_step_twelve_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self._stop_event.clear()
        self.stop_button.set_sensitive(True)
        self.step_twelve_row.set_sensitive(False)
        self._start_step(self.step_twelve_row)
        threading.Thread(target=self._run_step_twelve, daemon=True).start()

    def _run_all_steps(self) -> None:
        root_dir = self._resolve_case_root()
        self._run_steps_from_index(0, root_dir)

    def _run_steps_from_index(self, start_index: int, root_dir: Path | None) -> None:
        steps = self._pipeline_steps()
        success = True
        failed_step_id: str | None = None
        current_step_id: str | None = None
        try:
            for step_id, row, handler in steps[start_index:]:
                self._raise_if_stop_requested()
                current_step_id = step_id
                GLib.idle_add(self._start_step, row)
                if not handler():
                    success = False
                    failed_step_id = step_id
                    break
        except StopRequested:
            success = False
            if current_step_id and not failed_step_id:
                failed_step_id = current_step_id
        except Exception as exc:
            success = False
            if current_step_id and not failed_step_id:
                failed_step_id = current_step_id
            GLib.idle_add(self.show_toast, f"Pipeline failed: {exc}")
        finally:
            if failed_step_id and root_dir and root_dir.exists():
                self._safe_update_manifest(root_dir, {"last_failed_step": failed_step_id})
            GLib.idle_add(self._finish_run_all, success)

    def _finish_run_all(self, success: bool) -> None:
        stop_requested = self._stop_event.is_set()
        self._stop_event.clear()
        self._pipeline_running = False
        self.run_all_button.set_sensitive(True)
        self.stop_button.set_sensitive(False)
        self.resume_button.set_sensitive(True)
        self.step_list.set_sensitive(True)
        self._stop_status()
        if stop_requested:
            self.show_toast("Pipeline stopped.")
        elif success:
            self.show_toast("Pipeline complete.")
        else:
            self.show_toast("Pipeline stopped. Fix the errors and try again.")

    def _run_step_two(self) -> bool:
        success: bool | None = False
        try:
            self._raise_if_stop_requested()
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "case_bundle"
            text_dir = root_dir / "text_pages"
            if not text_dir.exists():
                raise FileNotFoundError("Run Create files to generate text files first.")
            shared_settings = load_classifier_settings()
            if (
                not shared_settings["api_url"]
                or not shared_settings["model_id"]
                or not shared_settings["api_key"]
            ):
                raise ValueError("Configure classify API URL, model ID, and API key in Settings.")
            classification_dir = root_dir / "classification"
            classification_dir.mkdir(parents=True, exist_ok=True)
            classify_basic_path = classification_dir / "basic.jsonl"
            classify_basic_path.touch(exist_ok=True)
            done_basic = _load_jsonl_file_names(classify_basic_path)
            text_files = sorted(text_dir.glob("*.txt"), key=_natural_sort_key)
            if not text_files:
                raise FileNotFoundError("No text files found to classify.")
            basic_settings = {
                "api_url": shared_settings["api_url"],
                "model_id": shared_settings["model_id"],
                "api_key": shared_settings["api_key"],
                "prompt": shared_settings["prompt"],
            }
            for text_path in text_files:
                self._raise_if_stop_requested()
                if text_path.name in done_basic:
                    continue
                content = text_path.read_text(encoding="utf-8", errors="ignore")
                entry = self._classify_text(basic_settings, text_path.name, content)
                with classify_basic_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry))
                    handle.write("\n")
                done_basic.add(text_path.name)
        except StopRequested:
            success = None
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Classification basic failed: {exc}")
        else:
            success = True
            self._safe_update_manifest(
                root_dir,
                {
                    "last_completed_step": "classify_basic",
                    "last_failed_step": None,
                    "last_failed_at": None,
                },
            )
            GLib.idle_add(self.show_toast, "Classification basic complete.")
        finally:
            GLib.idle_add(self.step_two_row.set_sensitive, True)
            GLib.idle_add(self._finish_step, self.step_two_row, success)
            GLib.idle_add(self._stop_status_if_idle)
            GLib.idle_add(self._stop_button_if_idle)
        return success is True

    def _run_step_correct_basic(self) -> bool:
        success: bool | None = False
        try:
            self._raise_if_stop_requested()
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "case_bundle"
            classification_dir = root_dir / "classification"
            classify_basic_path = classification_dir / "basic.jsonl"
            if not classify_basic_path.exists():
                raise FileNotFoundError(
                    "Run Classification basic to generate basic classifications first."
                )
            entries = _load_jsonl_entries(classify_basic_path)
            if not entries:
                raise FileNotFoundError("No entries found in basic.jsonl.")
            page_types: list[str] = [
                _extract_entry_value(entry, "page_type", "pagetype").strip().lower()
                for entry in entries
            ]
            corrected = list(page_types)
            total = len(corrected)

            def is_hearing(value: str) -> bool:
                return value == "hearing"

            def is_hearing_like(value: str) -> bool:
                return value in {"hearing", "hearing_first_page"}

            def is_report(value: str) -> bool:
                return value == "report"

            def is_cover_or_index(value: str) -> bool:
                return value in {"cover", "index"}

            index = 0
            while index < total:
                self._raise_if_stop_requested()
                if index + 1 < total and not is_hearing_like(
                    corrected[index]
                ) and not is_hearing_like(corrected[index + 1]):
                    if (
                        index - 2 >= 0
                        and index + 3 < total
                        and all(
                            is_hearing_like(corrected[i]) for i in range(index - 2, index)
                        )
                        and all(
                            is_hearing_like(corrected[i])
                            for i in range(index + 2, index + 4)
                        )
                    ):
                        corrected[index] = "hearing"
                        corrected[index + 1] = "hearing"
                        index += 2
                        continue
                if not is_hearing_like(corrected[index]):
                    if (
                        index - 2 >= 0
                        and index + 2 < total
                        and all(
                            is_hearing_like(corrected[i]) for i in range(index - 2, index)
                        )
                        and all(
                            is_hearing_like(corrected[i])
                            for i in range(index + 1, index + 3)
                        )
                    ):
                        corrected[index] = "hearing"
                index += 1

            for index in range(1, total - 1):
                self._raise_if_stop_requested()
                if (
                    corrected[index] == "hearing_first_page"
                    and is_cover_or_index(corrected[index - 1])
                    and is_cover_or_index(corrected[index + 1])
                ):
                    corrected[index] = "cover"

            index = 0
            while index < total:
                self._raise_if_stop_requested()
                if corrected[index] != "hearing":
                    index += 1
                    continue
                start = index
                while index < total and corrected[index] == "hearing":
                    index += 1
                if index - start >= 2 and start > 0 and is_cover_or_index(corrected[start - 1]):
                    corrected[start] = "hearing_first_page"

            for index in range(2, total - 2):
                self._raise_if_stop_requested()
                if not is_report(corrected[index]):
                    if (
                        is_report(corrected[index - 1])
                        and is_report(corrected[index - 2])
                        and is_report(corrected[index + 1])
                        and is_report(corrected[index + 2])
                    ):
                        corrected[index] = "report"

            for index in range(1, total - 1):
                self._raise_if_stop_requested()
                if (
                    corrected[index] == "minute_order"
                    and corrected[index - 1] not in {"minute_order", "minute_order_first_page"}
                    and corrected[index + 1] not in {"minute_order", "minute_order_first_page"}
                ):
                    corrected[index] = "minute_order_first_page"

            index = 0
            while index < total:
                self._raise_if_stop_requested()
                if corrected[index] != "minute_order":
                    index += 1
                    continue
                start = index
                while index < total and corrected[index] == "minute_order":
                    index += 1
                if index - start >= 2:
                    prev = corrected[start - 1] if start > 0 else ""
                    if prev not in {"minute_order", "minute_order_first_page"}:
                        corrected[start] = "minute_order_first_page"

            corrections = 0
            for entry, new_type, old_type in zip(entries, corrected, page_types):
                if new_type and new_type != old_type:
                    entry["page_type"] = new_type
                    corrections += 1

            corrected_path = classification_dir / "basic_corrected.jsonl"
            with corrected_path.open("w", encoding="utf-8") as handle:
                for entry in entries:
                    handle.write(json.dumps(entry))
                    handle.write("\n")
        except StopRequested:
            success = None
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Correct basic classification failed: {exc}")
        else:
            success = True
            self._safe_update_manifest(
                root_dir,
                {
                    "last_completed_step": "correct_basic",
                    "last_failed_step": None,
                    "last_failed_at": None,
                },
            )
            GLib.idle_add(
                self.show_toast,
                f"Correct basic classification complete. {corrections} changes applied.",
            )
        finally:
            GLib.idle_add(self.step_correct_basic_row.set_sensitive, True)
            GLib.idle_add(self._finish_step, self.step_correct_basic_row, success)
            GLib.idle_add(self._stop_status_if_idle)
            GLib.idle_add(self._stop_button_if_idle)
        return success is True

    def _run_step_filter(self) -> bool:
        success: bool | None = False
        try:
            self._raise_if_stop_requested()
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "case_bundle"
            text_dir = root_dir / "text_pages"
            if not text_dir.exists():
                raise FileNotFoundError("Run Create files to generate text files first.")
            classification_dir = root_dir / "classification"
            classify_basic_path = classification_dir / "basic_corrected.jsonl"
            if not classify_basic_path.exists():
                raise FileNotFoundError(
                    "Run Correct basic classification to generate corrected classifications first."
                )
            filter_settings = load_form_filter_settings()
            if (
                not filter_settings["api_url"]
                or not filter_settings["model_id"]
                or not filter_settings["api_key"]
            ):
                raise ValueError(
                    "Configure classification filter API URL, model ID, and API key in Settings."
                )
            classification_dir.mkdir(parents=True, exist_ok=True)
            relevant_forms_path = classification_dir / "relevant_forms.jsonl"
            relevant_forms_path.touch(exist_ok=True)
            done_relevant = _load_jsonl_file_names(relevant_forms_path)
            form_targets = _load_classify_form_targets(classify_basic_path)
            if form_targets:
                filter_payload = {
                    "api_url": filter_settings["api_url"],
                    "model_id": filter_settings["model_id"],
                    "api_key": filter_settings["api_key"],
                    "prompt": filter_settings["prompt"],
                }
                for file_name, _page_type in form_targets:
                    self._raise_if_stop_requested()
                    if file_name in done_relevant:
                        continue
                    text_path = text_dir / file_name
                    if not text_path.exists():
                        raise FileNotFoundError(
                            f"Missing text file {file_name} for form filter classification."
                        )
                    content = text_path.read_text(encoding="utf-8", errors="ignore")
                    entry = self._classify_text(filter_payload, file_name, content)
                    is_relevant = _is_truthy(_extract_entry_value(entry, "is_relevant", "relevant", "keep"))
                    if not is_relevant:
                        continue
                    ordered_entry: dict[str, str | int] = {"file_name": file_name}
                    page_number = _extract_page_number(file_name)
                    if page_number is not None:
                        ordered_entry["page_number"] = page_number
                    with relevant_forms_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(ordered_entry))
                        handle.write("\n")
                    done_relevant.add(file_name)
            relevant_reports_path = classification_dir / "relevant_reports.jsonl"
            relevant_reports_path.touch(exist_ok=True)
            done_reports = _load_jsonl_file_names(relevant_reports_path)
            report_targets = _load_classify_report_targets(classify_basic_path)
            if report_targets:
                report_payload = {
                    "api_url": filter_settings["api_url"],
                    "model_id": filter_settings["model_id"],
                    "api_key": filter_settings["api_key"],
                    "prompt": filter_settings["report_prompt"],
                }
                for file_name, _page_type in report_targets:
                    self._raise_if_stop_requested()
                    if file_name in done_reports:
                        continue
                    text_path = text_dir / file_name
                    if not text_path.exists():
                        raise FileNotFoundError(
                            f"Missing text file {file_name} for report filter classification."
                        )
                    content = text_path.read_text(encoding="utf-8", errors="ignore")
                    entry = self._classify_text(report_payload, file_name, content)
                    is_relevant = _is_truthy(_extract_entry_value(entry, "is_relevant", "relevant", "keep"))
                    if not is_relevant:
                        continue
                    ordered_entry = {"file_name": file_name}
                    page_number = _extract_page_number(file_name)
                    if page_number is not None:
                        ordered_entry["page_number"] = page_number
                    with relevant_reports_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(ordered_entry))
                        handle.write("\n")
                    done_reports.add(file_name)
        except StopRequested:
            success = None
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Classification filter failed: {exc}")
        else:
            success = True
            self._safe_update_manifest(
                root_dir,
                {
                    "last_completed_step": "classify_filter",
                    "last_failed_step": None,
                    "last_failed_at": None,
                },
            )
            GLib.idle_add(self.show_toast, "Classification filter complete.")
        finally:
            GLib.idle_add(self.step_filter_row.set_sensitive, True)
            GLib.idle_add(self._finish_step, self.step_filter_row, success)
            GLib.idle_add(self._stop_status_if_idle)
            GLib.idle_add(self._stop_button_if_idle)
        return success is True

    def _run_step_three(self) -> bool:
        success: bool | None = False
        try:
            self._raise_if_stop_requested()
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "case_bundle"
            text_dir = root_dir / "text_pages"
            if not text_dir.exists():
                raise FileNotFoundError("Run Create files to generate text files first.")
            classification_dir = root_dir / "classification"
            classify_basic_path = classification_dir / "basic_corrected.jsonl"
            if not classify_basic_path.exists():
                raise FileNotFoundError(
                    "Run Correct basic classification to generate corrected classifications first."
                )
            relevant_forms_path = classification_dir / "relevant_forms.jsonl"
            if not relevant_forms_path.exists():
                raise FileNotFoundError(
                    "Run Classification filter to generate relevant form classifications first."
                )
            relevant_reports_path = classification_dir / "relevant_reports.jsonl"
            if not relevant_reports_path.exists():
                raise FileNotFoundError(
                    "Run Classification filter to generate relevant report classifications first."
                )
            further_settings = load_classify_further_settings()
            if (
                not further_settings["api_url"]
                or not further_settings["model_id"]
                or not further_settings["api_key"]
            ):
                raise ValueError(
                    "Configure classify further API URL, model ID, and API key in Settings."
                )

            classify_dates_path = classification_dir / "dates.jsonl"
            classify_dates_path.touch(exist_ok=True)
            done_dates = _load_jsonl_file_names(classify_dates_path)
            date_targets = _load_classify_date_targets(classify_basic_path)
            if date_targets:
                date_settings = {
                    "api_url": further_settings["api_url"],
                    "model_id": further_settings["model_id"],
                    "api_key": further_settings["api_key"],
                    "prompt": further_settings["dates_prompt"],
                }
                for file_name, _page_type in date_targets:
                    self._raise_if_stop_requested()
                    if file_name in done_dates:
                        continue
                    text_path = text_dir / file_name
                    if not text_path.exists():
                        raise FileNotFoundError(
                            f"Missing text file {file_name} for date classification."
                        )
                    content = text_path.read_text(encoding="utf-8", errors="ignore")
                    entry = self._classify_text(date_settings, file_name, content)
                    ordered_entry: dict[str, str] = {"file_name": file_name}
                    for key, value in entry.items():
                        if key == "file_name":
                            continue
                        ordered_entry[key] = value
                    with classify_dates_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(ordered_entry))
                        handle.write("\n")
                    done_dates.add(file_name)

            classify_report_names_path = classification_dir / "report_names.jsonl"
            classify_report_names_path.touch(exist_ok=True)
            done_reports = _load_jsonl_file_names(classify_report_names_path)
            report_targets = _load_relevant_report_targets(relevant_reports_path)
            if report_targets:
                report_settings_payload = {
                    "api_url": further_settings["api_url"],
                    "model_id": further_settings["model_id"],
                    "api_key": further_settings["api_key"],
                    "prompt": further_settings["reports_prompt"],
                }
                for file_name in report_targets:
                    self._raise_if_stop_requested()
                    if file_name in done_reports:
                        continue
                    text_path = text_dir / file_name
                    if not text_path.exists():
                        raise FileNotFoundError(
                            f"Missing text file {file_name} for report name classification."
                        )
                    content = text_path.read_text(encoding="utf-8", errors="ignore")
                    entry = self._classify_text(report_settings_payload, file_name, content)
                    ordered_entry: dict[str, str] = {"file_name": file_name}
                    for key, value in entry.items():
                        if key == "file_name":
                            continue
                        ordered_entry[key] = value
                    with classify_report_names_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(ordered_entry))
                        handle.write("\n")
                    done_reports.add(file_name)

            classify_form_names_path = classification_dir / "form_names.jsonl"
            classify_form_names_path.touch(exist_ok=True)
            done_forms = _load_jsonl_file_names(classify_form_names_path)
            form_targets = _load_relevant_form_targets(relevant_forms_path)
            if form_targets:
                form_settings_payload = {
                    "api_url": further_settings["api_url"],
                    "model_id": further_settings["model_id"],
                    "api_key": further_settings["api_key"],
                    "prompt": further_settings["forms_prompt"],
                }
                for file_name in form_targets:
                    self._raise_if_stop_requested()
                    if file_name in done_forms:
                        continue
                    text_path = text_dir / file_name
                    if not text_path.exists():
                        raise FileNotFoundError(
                            f"Missing text file {file_name} for form name classification."
                        )
                    content = text_path.read_text(encoding="utf-8", errors="ignore")
                    entry = self._classify_text(form_settings_payload, file_name, content)
                    ordered_entry: dict[str, str] = {"file_name": file_name}
                    for key, value in entry.items():
                        if key == "file_name":
                            continue
                        ordered_entry[key] = value
                    with classify_form_names_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(ordered_entry))
                        handle.write("\n")
                    done_forms.add(file_name)
        except StopRequested:
            success = None
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Classification dates/names failed: {exc}")
        else:
            success = True
            self._safe_update_manifest(
                root_dir,
                {
                    "last_completed_step": "classify_further",
                    "last_failed_step": None,
                    "last_failed_at": None,
                },
            )
            GLib.idle_add(self.show_toast, "Classification dates/names complete.")
        finally:
            GLib.idle_add(self.step_three_row.set_sensitive, True)
            GLib.idle_add(self._finish_step, self.step_three_row, success)
            GLib.idle_add(self._stop_status_if_idle)
            GLib.idle_add(self._stop_button_if_idle)
        return success is True

    def _run_step_six(self) -> bool:
        success: bool | None = False
        try:
            self._raise_if_stop_requested()
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "case_bundle"
            classification_dir = root_dir / "classification"
            derived_dir = root_dir / "artifacts"
            classify_basic_path = classification_dir / "basic_corrected.jsonl"
            classify_dates_path = classification_dir / "dates.jsonl"
            classify_report_names_path = classification_dir / "report_names.jsonl"
            classify_form_names_path = classification_dir / "form_names.jsonl"
            for path in (
                classify_basic_path,
                classify_dates_path,
                classify_report_names_path,
                classify_form_names_path,
            ):
                if not path.exists():
                    raise FileNotFoundError(
                        "Run Correct basic classification, filter, and dates/names to generate classify JSONL files first."
                    )
            derived_dir.mkdir(parents=True, exist_ok=True)
            date_entries = _load_jsonl_entries(classify_dates_path)
            basic_entries = _load_jsonl_entries(classify_basic_path)
            date_by_file: dict[str, str] = {}
            for entry in date_entries:
                self._raise_if_stop_requested()
                file_name = _extract_entry_value(entry, "file_name", "filename")
                if not file_name:
                    continue
                date_value = _extract_entry_value(entry, "date")
                date_by_file[file_name] = date_value
            for entry in basic_entries:
                self._raise_if_stop_requested()
                file_name = _extract_entry_value(entry, "file_name", "filename")
                if not file_name or file_name in date_by_file:
                    continue
                date_value = _extract_entry_value(entry, "date")
                if date_value:
                    date_by_file[file_name] = date_value
            form_lines: list[str] = []
            for entry in _load_jsonl_entries(classify_form_names_path):
                self._raise_if_stop_requested()
                form_name = _extract_entry_value(entry, "form_name", "form")
                if not form_name:
                    continue
                page = _page_label_from_filename(
                    _extract_entry_value(entry, "file_name", "filename")
                )
                form_lines.append(_format_toc_line(form_name, page))
            report_lines: list[str] = []
            for entry in _load_jsonl_entries(classify_report_names_path):
                self._raise_if_stop_requested()
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
                self._raise_if_stop_requested()
                date_value = date_by_file.get(file_name, "").strip()
                page = _page_label_from_filename(file_name)
                line = _format_toc_line(date_value, page)
                if page_type == "minute_order_first_page":
                    minute_order_lines.append(line)
                elif page_type == "hearing_first_page":
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
            toc_path = derived_dir / "toc.txt"
            toc_path.write_text("\n".join(toc_lines).rstrip() + "\n", encoding="utf-8")
        except StopRequested:
            success = None
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Build TOC failed: {exc}")
        else:
            success = True
            self._safe_update_manifest(
                root_dir,
                {
                    "last_completed_step": "build_toc",
                    "last_failed_step": None,
                    "last_failed_at": None,
                },
            )
            GLib.idle_add(self.show_toast, "Build TOC complete.")
        finally:
            GLib.idle_add(self.step_six_row.set_sensitive, True)
            GLib.idle_add(self._finish_step, self.step_six_row, success)
            GLib.idle_add(self._stop_status_if_idle)
            GLib.idle_add(self._stop_button_if_idle)
        return success is True

    def _run_step_seven(self) -> bool:
        success: bool | None = False
        try:
            self._raise_if_stop_requested()
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "case_bundle"
            classification_dir = root_dir / "classification"
            derived_dir = root_dir / "artifacts"
            classify_basic_path = classification_dir / "basic_corrected.jsonl"
            classify_dates_path = classification_dir / "dates.jsonl"
            classify_report_names_path = classification_dir / "report_names.jsonl"
            relevant_reports_path = classification_dir / "relevant_reports.jsonl"
            for path in (classify_basic_path, classify_dates_path, classify_report_names_path):
                if not path.exists():
                    raise FileNotFoundError(
                        "Run Correct basic classification, filter, and dates/names to generate classify JSONL files first."
                    )
            if not relevant_reports_path.exists():
                raise FileNotFoundError(
                    "Run Classification filter to generate relevant report classifications first."
                )
            derived_dir.mkdir(parents=True, exist_ok=True)
            date_by_file: dict[str, str] = {}
            for entry in _load_jsonl_entries(classify_dates_path):
                self._raise_if_stop_requested()
                file_name = _extract_entry_value(entry, "file_name", "filename")
                if not file_name:
                    continue
                date_value = _extract_entry_value(entry, "date")
                if date_value:
                    date_by_file[file_name] = date_value
            report_name_by_file: dict[str, str] = {}
            for entry in _load_jsonl_entries(classify_report_names_path):
                self._raise_if_stop_requested()
                file_name = _extract_entry_value(entry, "file_name", "filename")
                report_name = _extract_entry_value(entry, "report_name", "report", "name")
                if file_name and report_name:
                    report_name_by_file[file_name] = report_name
            relevant_report_files = set(_load_relevant_report_targets(relevant_reports_path))
            hearing_boundaries: list[dict[str, str]] = []
            report_boundaries: list[dict[str, str]] = []
            minutes_boundaries: list[dict[str, str]] = []
            entries = _load_classify_basic_entries(classify_basic_path)
            if not entries:
                raise FileNotFoundError("No entries found in basic_corrected.jsonl.")
            current_report_start: str | None = None
            current_report_end: str | None = None
            report_sequence_relevant = False
            for file_name, page_type, page_number in entries:
                self._raise_if_stop_requested()
                if page_type != "report":
                    if current_report_start:
                        if report_sequence_relevant:
                            self._append_boundary_entry(
                                "report",
                                current_report_start,
                                current_report_end,
                                date_by_file,
                                report_name_by_file,
                                hearing_boundaries,
                                report_boundaries,
                                minutes_boundaries,
                            )
                        current_report_start = None
                        current_report_end = None
                        report_sequence_relevant = False
                    continue
                if (
                    current_report_end is not None
                    and _extract_page_number(current_report_end) == page_number - 1
                ):
                    current_report_end = file_name
                else:
                    if current_report_start:
                        if report_sequence_relevant:
                            self._append_boundary_entry(
                                "report",
                                current_report_start,
                                current_report_end,
                                date_by_file,
                                report_name_by_file,
                                hearing_boundaries,
                                report_boundaries,
                                minutes_boundaries,
                            )
                    current_report_start = file_name
                    current_report_end = file_name
                    report_sequence_relevant = file_name in relevant_report_files
            if current_report_start:
                if report_sequence_relevant:
                    self._append_boundary_entry(
                        "report",
                        current_report_start,
                        current_report_end,
                        date_by_file,
                        report_name_by_file,
                        hearing_boundaries,
                        report_boundaries,
                        minutes_boundaries,
                    )

            index = 0
            total = len(entries)
            while index < total:
                self._raise_if_stop_requested()
                file_name, page_type, page_number = entries[index]
                if page_type in {"hearing_first_page", "minute_order_first_page"}:
                    expected_follow_type = (
                        "hearing"
                        if page_type == "hearing_first_page"
                        else "minute_order"
                    )
                    end_file = file_name
                    last_number = page_number
                    index += 1
                    while index < total:
                        self._raise_if_stop_requested()
                        next_file, next_type, next_number = entries[index]
                        if (
                            next_type != expected_follow_type
                            or next_number != last_number + 1
                        ):
                            break
                        end_file = next_file
                        last_number = next_number
                        index += 1
                    self._append_boundary_entry(
                        page_type,
                        file_name,
                        end_file,
                        date_by_file,
                        report_name_by_file,
                        hearing_boundaries,
                        report_boundaries,
                        minutes_boundaries,
                    )
                    continue
                index += 1
            hearing_path = derived_dir / "hearing_boundaries.json"
            hearing_path.write_text(
                json.dumps(hearing_boundaries, indent=2),
                encoding="utf-8",
            )
            report_path = derived_dir / "report_boundaries.json"
            report_path.write_text(
                json.dumps(report_boundaries, indent=2),
                encoding="utf-8",
            )
            minutes_path = derived_dir / "minutes_boundaries.json"
            minutes_path.write_text(
                json.dumps(minutes_boundaries, indent=2),
                encoding="utf-8",
            )
        except StopRequested:
            success = None
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Find boundaries failed: {exc}")
        else:
            success = True
            self._safe_update_manifest(
                root_dir,
                {
                    "last_completed_step": "find_boundaries",
                    "last_failed_step": None,
                    "last_failed_at": None,
                },
            )
            GLib.idle_add(self.show_toast, "Find boundaries complete.")
        finally:
            GLib.idle_add(self.step_seven_row.set_sensitive, True)
            GLib.idle_add(self._finish_step, self.step_seven_row, success)
            GLib.idle_add(self._stop_status_if_idle)
            GLib.idle_add(self._stop_button_if_idle)
        return success is True

    def _run_step_eight(self) -> bool:
        success: bool | None = False
        try:
            self._raise_if_stop_requested()
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "case_bundle"
            derived_dir = root_dir / "artifacts"
            text_dir = root_dir / "text_pages"
            if not text_dir.exists():
                raise FileNotFoundError("Run Create files to generate text files first.")
            hearing_path = derived_dir / "hearing_boundaries.json"
            report_path = derived_dir / "report_boundaries.json"
            if not hearing_path.exists() or not report_path.exists():
                raise FileNotFoundError("Run Find boundaries to generate boundary JSON files first.")
            artifacts_dir = root_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
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
            (artifacts_dir / "raw_hearings.txt").write_text(raw_hearings, encoding="utf-8")
            (artifacts_dir / "raw_reports.txt").write_text(raw_reports, encoding="utf-8")
        except StopRequested:
            success = None
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Create raw failed: {exc}")
        else:
            success = True
            self._safe_update_manifest(
                root_dir,
                {
                    "last_completed_step": "create_raw",
                    "last_failed_step": None,
                    "last_failed_at": None,
                },
            )
            GLib.idle_add(self.show_toast, "Create raw complete.")
        finally:
            GLib.idle_add(self.step_eight_row.set_sensitive, True)
            GLib.idle_add(self._finish_step, self.step_eight_row, success)
            GLib.idle_add(self._stop_status_if_idle)
            GLib.idle_add(self._stop_button_if_idle)
        return success is True

    def _run_step_nine(self) -> bool:
        success: bool | None = False
        try:
            self._raise_if_stop_requested()
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "case_bundle"
            artifacts_dir = root_dir / "artifacts"
            raw_hearings_path = artifacts_dir / "raw_hearings.txt"
            raw_reports_path = artifacts_dir / "raw_reports.txt"
            if not raw_hearings_path.exists() or not raw_reports_path.exists():
                raise FileNotFoundError("Run Create raw to generate raw hearing/report files first.")
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

            checkpoint_dir = _checkpoint_dir(root_dir)
            hearing_checkpoint_path = checkpoint_dir / "optimize_hearings.jsonl"
            report_checkpoint_path = checkpoint_dir / "optimize_reports.jsonl"
            hearing_checkpoint = _load_indexed_jsonl(hearing_checkpoint_path)
            report_checkpoint = _load_indexed_jsonl(report_checkpoint_path)

            optimized_hearings: list[str] = []
            hearing_index = 0
            for label, content in hearing_sections:
                self._raise_if_stop_requested()
                if not content:
                    continue
                sentences = _split_into_sentences(content)
                if not sentences:
                    continue
                chunks = _chunk_sentences(sentences, 3500)
                if not chunks:
                    continue
                needs_work = any(
                    (hearing_index + offset) not in hearing_checkpoint
                    for offset in range(len(chunks))
                )
                attorney_info = ""
                if needs_work:
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
                for offset, chunk in enumerate(chunks):
                    self._raise_if_stop_requested()
                    index = hearing_index + offset
                    if index in hearing_checkpoint:
                        continue
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
                    cleaned_response = response.strip() if response else ""
                    hearing_checkpoint[index] = cleaned_response
                    _append_indexed_jsonl(hearing_checkpoint_path, index, cleaned_response, label=label)
                hearing_index += len(chunks)

            for index in range(hearing_index):
                self._raise_if_stop_requested()
                response_text = hearing_checkpoint.get(index, "")
                if response_text:
                    optimized_hearings.append(response_text)

            optimized_reports: list[str] = []
            report_index = 0
            for label, content in report_sections:
                self._raise_if_stop_requested()
                if not content:
                    continue
                sentences = _split_into_sentences(content)
                if not sentences:
                    continue
                chunks = _chunk_sentences(sentences, 3500)
                if not chunks:
                    continue
                for offset, chunk in enumerate(chunks):
                    self._raise_if_stop_requested()
                    index = report_index + offset
                    if index in report_checkpoint:
                        continue
                    response = self._request_plain_text(
                        {
                            "api_url": settings["api_url"],
                            "model_id": settings["model_id"],
                            "api_key": settings["api_key"],
                            "prompt": reports_prompt,
                        },
                        chunk,
                    )
                    cleaned_response = response.strip() if response else ""
                    report_checkpoint[index] = cleaned_response
                    _append_indexed_jsonl(report_checkpoint_path, index, cleaned_response, label=label)
                report_index += len(chunks)

            for index in range(report_index):
                self._raise_if_stop_requested()
                response_text = report_checkpoint.get(index, "")
                if response_text:
                    optimized_reports.append(response_text)

            (artifacts_dir / "optimized_hearings.txt").write_text(
                _collapse_blank_lines("\n\n".join(optimized_hearings)),
                encoding="utf-8",
            )
            (artifacts_dir / "optimized_reports.txt").write_text(
                _collapse_blank_lines("\n\n".join(optimized_reports)),
                encoding="utf-8",
            )
        except StopRequested:
            success = None
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Create optimized failed: {exc}")
        else:
            success = True
            self._safe_update_manifest(
                root_dir,
                {
                    "last_completed_step": "create_optimized",
                    "last_failed_step": None,
                    "last_failed_at": None,
                },
            )
            GLib.idle_add(self.show_toast, "Create optimized complete.")
        finally:
            GLib.idle_add(self.step_nine_row.set_sensitive, True)
            GLib.idle_add(self._finish_step, self.step_nine_row, success)
            GLib.idle_add(self._stop_status_if_idle)
            GLib.idle_add(self._stop_button_if_idle)
        return success is True

    def _run_step_ten(self) -> bool:
        success: bool | None = False
        try:
            self._raise_if_stop_requested()
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "case_bundle"
            artifacts_dir = root_dir / "artifacts"
            summaries_dir = root_dir / "summaries"
            summaries_path, reports_path = _summary_output_paths(root_dir)
            minutes_path = _minutes_summary_output_path(root_dir)
            text_dir = root_dir / "text_pages"
            optimized_hearings_path = artifacts_dir / "optimized_hearings.txt"
            optimized_reports_path = artifacts_dir / "optimized_reports.txt"
            if not optimized_hearings_path.exists() or not optimized_reports_path.exists():
                raise FileNotFoundError("Run Create optimized to generate optimized files first.")
            if not text_dir.exists():
                raise FileNotFoundError("Run Create files to generate text files first.")
            minutes_boundaries_path = artifacts_dir / "minutes_boundaries.json"
            if not minutes_boundaries_path.exists():
                raise FileNotFoundError("Run Find boundaries to generate minute order boundaries first.")
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

            checkpoint_dir = _checkpoint_dir(root_dir)
            hearing_checkpoint_path = checkpoint_dir / "summarize_hearings.jsonl"
            report_checkpoint_path = checkpoint_dir / "summarize_reports.jsonl"
            minutes_checkpoint_path = checkpoint_dir / "summarize_minutes.jsonl"
            hearing_checkpoint = _load_indexed_jsonl(hearing_checkpoint_path)
            report_checkpoint = _load_indexed_jsonl(report_checkpoint_path)
            minutes_checkpoint = _load_indexed_jsonl(minutes_checkpoint_path)

            case_name, _root_dir = load_case_context()
            display_case_name = case_name.replace("_", " ") if case_name else ""
            summary_hearings: list[str] = []
            summary_reports: list[str] = []

            if display_case_name:
                summary_hearings.extend(["Hearings Summary", display_case_name, ""])
            else:
                summary_hearings.append("Hearings Summary")

            hearing_paragraphs = _split_paragraphs(
                optimized_hearings_path.read_text(encoding="utf-8")
            )
            hearing_groups: list[tuple[str, list[str]]] = []
            current_date: str | None = None
            for paragraph in hearing_paragraphs:
                self._raise_if_stop_requested()
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

            hearing_chunk_index = 0
            for date_value, paragraphs in hearing_groups:
                self._raise_if_stop_requested()
                for chunk in _chunk_paragraphs(paragraphs, chunk_size):
                    self._raise_if_stop_requested()
                    if hearing_chunk_index in hearing_checkpoint:
                        hearing_chunk_index += 1
                        continue
                    response = self._request_plain_text(
                        {
                            "api_url": settings["api_url"],
                            "model_id": settings["model_id"],
                            "api_key": settings["api_key"],
                            "prompt": settings["hearings_prompt"],
                        },
                        chunk,
                    )
                    cleaned_response = response.strip() if response else ""
                    hearing_checkpoint[hearing_chunk_index] = cleaned_response
                    _append_indexed_jsonl(
                        hearing_checkpoint_path,
                        hearing_chunk_index,
                        cleaned_response,
                        label=date_value or "HEARING",
                    )
                    hearing_chunk_index += 1

            first_section = True
            hearing_chunk_index = 0
            for date_value, paragraphs in hearing_groups:
                self._raise_if_stop_requested()
                if not first_section:
                    summary_hearings.append("")
                summary_hearings.append(date_value or "HEARING")
                summary_hearings.append("")
                first_section = False
                for chunk in _chunk_paragraphs(paragraphs, chunk_size):
                    self._raise_if_stop_requested()
                    response = hearing_checkpoint.get(hearing_chunk_index, "")
                    hearing_chunk_index += 1
                    if response:
                        cleaned_response = _remove_hearing_date_mentions(response.strip())
                        summary_hearings.append(_remove_standalone_date_lines(cleaned_response))
                        summary_hearings.append("")

            if display_case_name:
                summary_reports.extend(["Reports Summary", display_case_name, "", ""])
            else:
                summary_reports.extend(["Reports Summary", ""])

            report_paragraphs = _split_paragraphs(
                optimized_reports_path.read_text(encoding="utf-8")
            )
            report_paragraphs = [
                re.sub(r"^Reporting:\s*", "", paragraph) for paragraph in report_paragraphs
            ]
            report_chunk_index = 0
            for chunk in _chunk_paragraphs(report_paragraphs, chunk_size):
                self._raise_if_stop_requested()
                if report_chunk_index not in report_checkpoint:
                    response = self._request_plain_text(
                        {
                            "api_url": settings["api_url"],
                            "model_id": settings["model_id"],
                            "api_key": settings["api_key"],
                            "prompt": settings["reports_prompt"],
                        },
                        chunk,
                    )
                    cleaned_response = response.strip() if response else ""
                    report_checkpoint[report_chunk_index] = cleaned_response
                    _append_indexed_jsonl(
                        report_checkpoint_path,
                        report_chunk_index,
                        cleaned_response,
                    )
                report_chunk_index += 1

            report_chunk_index = 0
            for chunk in _chunk_paragraphs(report_paragraphs, chunk_size):
                self._raise_if_stop_requested()
                response = report_checkpoint.get(report_chunk_index, "")
                report_chunk_index += 1
                if response:
                    summary_reports.append(response.strip())
                    summary_reports.append("")

            minutes_outline: list[str] = []
            if display_case_name:
                minutes_outline.extend(["Minutes Summary", display_case_name, ""])
            else:
                minutes_outline.append("Minutes Summary")

            minute_entries = _load_json_entries(minutes_boundaries_path)
            minutes_index = 0
            for entry in minute_entries:
                self._raise_if_stop_requested()
                date_value = _extract_entry_value(entry, "date").strip()
                start_label = _extract_entry_value(entry, "start_page", "start", "starte_page").strip()
                end_label = _extract_entry_value(entry, "end_page", "end", "endpage").strip()
                start_page = _page_number_from_label(start_label)
                end_page = _page_number_from_label(end_label)
                if start_page is None or end_page is None:
                    raise ValueError("Minute order boundary entry missing start/end page.")
                if end_page < start_page:
                    raise ValueError("Minute order boundary entry has end page before start page.")
                minutes_outline.append(date_value or "Minute Order")
                minutes_outline.append("")
                page_texts: list[str] = []
                for page in range(start_page, end_page + 1):
                    self._raise_if_stop_requested()
                    page_path = text_dir / f"{page:04d}.txt"
                    if not page_path.exists():
                        raise FileNotFoundError(f"Missing text file {page_path.name}.")
                    page_texts.append(page_path.read_text(encoding="utf-8", errors="ignore"))
                minutes_payload = "\n".join(page_texts).strip()
                response = minutes_checkpoint.get(minutes_index, "")
                if minutes_index not in minutes_checkpoint:
                    if minutes_payload:
                        response = self._request_plain_text(
                            {
                                "api_url": settings["api_url"],
                                "model_id": settings["model_id"],
                                "api_key": settings["api_key"],
                                "prompt": settings["minutes_prompt"],
                            },
                            minutes_payload,
                        )
                    response = response.strip() if response else ""
                    minutes_checkpoint[minutes_index] = response
                    _append_indexed_jsonl(
                        minutes_checkpoint_path,
                        minutes_index,
                        response,
                        label=date_value or "Minute Order",
                    )
                if response:
                    minutes_outline.append(" ".join(response.split()))
                else:
                    minutes_outline.append("")
                minutes_outline.append("")
                minutes_index += 1

            summaries_dir.mkdir(parents=True, exist_ok=True)
            summaries_path.write_text(
                _collapse_blank_lines("\n".join(summary_hearings)),
                encoding="utf-8",
            )
            reports_path.write_text(
                _collapse_blank_lines("\n".join(summary_reports)),
                encoding="utf-8",
            )
            minutes_path.write_text(
                _collapse_blank_lines("\n".join(minutes_outline)),
                encoding="utf-8",
            )
        except StopRequested:
            success = None
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Create summaries failed: {exc}")
        else:
            success = True
            self._safe_update_manifest(
                root_dir,
                {
                    "last_completed_step": "create_summaries",
                    "last_failed_step": None,
                    "last_failed_at": None,
                },
            )
            GLib.idle_add(self.show_toast, "Create summaries complete.")
        finally:
            GLib.idle_add(self.step_ten_row.set_sensitive, True)
            GLib.idle_add(self._finish_step, self.step_ten_row, success)
            GLib.idle_add(self._stop_status_if_idle)
            GLib.idle_add(self._stop_button_if_idle)
        return success is True

    def _run_step_eleven(self) -> bool:
        success: bool | None = False
        try:
            self._raise_if_stop_requested()
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "case_bundle"
            summaries_dir = root_dir / "summaries"
            summaries_path, reports_path = _summary_output_paths(root_dir)
            if not summaries_path.exists() or not reports_path.exists():
                raise FileNotFoundError("Run Create summaries to generate summarized files first.")
            settings = load_overview_settings()
            if not settings["api_url"] or not settings["model_id"] or not settings["api_key"]:
                raise ValueError("Configure overview API URL, model ID, and API key in Settings.")
            hearings_text = summaries_path.read_text(encoding="utf-8", errors="ignore")
            reports_text = reports_path.read_text(encoding="utf-8", errors="ignore")
            combined = "\n\n".join(
                [
                    "Summarized Hearings:",
                    hearings_text.strip(),
                    "",
                    "Summarized Reports:",
                    reports_text.strip(),
                ]
            ).strip()
            overview = self._request_plain_text(
                {
                    "api_url": settings["api_url"],
                    "model_id": settings["model_id"],
                    "api_key": settings["api_key"],
                    "prompt": settings["prompt"],
                },
                combined,
            )
            if not overview:
                raise ValueError("Overview response was empty.")
            rag_dir = root_dir / "rag"
            rag_dir.mkdir(parents=True, exist_ok=True)
            (rag_dir / "case_overview.txt").write_text(
                _collapse_blank_lines(overview),
                encoding="utf-8",
            )
        except StopRequested:
            success = None
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Case overview failed: {exc}")
        else:
            success = True
            self._safe_update_manifest(
                root_dir,
                {
                    "last_completed_step": "case_overview",
                    "last_failed_step": None,
                    "last_failed_at": None,
                },
            )
            GLib.idle_add(self.show_toast, "Case overview complete.")
        finally:
            GLib.idle_add(self.step_eleven_row.set_sensitive, True)
            GLib.idle_add(self._finish_step, self.step_eleven_row, success)
            GLib.idle_add(self._stop_status_if_idle)
            GLib.idle_add(self._stop_button_if_idle)
        return success is True

    def _run_step_twelve(self) -> bool:
        success: bool | None = False
        try:
            self._raise_if_stop_requested()
            parents = {path.parent for path in self.selected_pdfs}
            if len(parents) != 1:
                raise ValueError("Selected PDFs must be in the same folder.")
            base_dir = parents.pop()
            root_dir = base_dir / "case_bundle"
            summaries_dir = root_dir / "summaries"
            summaries_path, reports_path = _summary_output_paths(root_dir)
            if not summaries_path.exists() or not reports_path.exists():
                raise FileNotFoundError("Run Create summaries to generate summarized files first.")
            settings = load_rag_settings()
            if not settings["voyage_api_key"] or not settings["voyage_model"]:
                raise ValueError("Configure Voyage credentials in Settings.")
            try:
                from langchain_chroma import Chroma  # type: ignore
                from langchain_core.documents import Document  # type: ignore
                voyage_module = importlib.import_module("langchain_voyageai")
                rag_embedder_class = getattr(
                    voyage_module,
                    "VoyageAI" + "Emb" + "eddings",
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Missing langchain/chroma/voyage dependencies. See uv add instructions."
                ) from exc

            rag_dir = root_dir / "rag"
            vector_dir = rag_dir / "vector_database"
            vector_dir.mkdir(parents=True, exist_ok=True)

            rag_embedder = rag_embedder_class(
                voyage_api_key=settings["voyage_api_key"],
                model=settings["voyage_model"],
            )
            vectorstore = Chroma(
                persist_directory=str(vector_dir),
                embedding_function=rag_embedder,
            )

            hearing_text = summaries_path.read_text(encoding="utf-8", errors="ignore")
            report_text = reports_path.read_text(encoding="utf-8", errors="ignore")
            if not hearing_text.strip() and not report_text.strip():
                raise ValueError("No optimized content available for RAG index.")

            documents: list[Document] = []
            for paragraph in _split_paragraphs(hearing_text):
                self._raise_if_stop_requested()
                documents.append(
                    Document(
                        page_content=paragraph,
                        metadata={"source": summaries_path.name},
                    )
                )
            for paragraph in _split_paragraphs(report_text):
                self._raise_if_stop_requested()
                documents.append(
                    Document(
                        page_content=paragraph,
                        metadata={"source": reports_path.name},
                    )
                )
            if not documents:
                raise ValueError("No paragraphs found to embed.")
            vectorstore.add_documents(documents)

        except StopRequested:
            success = None
        except Exception as exc:
            GLib.idle_add(self.show_toast, f"Create RAG index failed: {exc}")
        else:
            success = True
            self._safe_update_manifest(
                root_dir,
                {
                    "last_completed_step": "create_rag_index",
                    "last_failed_step": None,
                    "last_failed_at": None,
                },
            )
            GLib.idle_add(self.show_toast, "Create RAG index complete.")
        finally:
            GLib.idle_add(self.step_twelve_row.set_sensitive, True)
            GLib.idle_add(self._finish_step, self.step_twelve_row, success)
            GLib.idle_add(self._stop_status_if_idle)
            GLib.idle_add(self._stop_button_if_idle)
        return success is True

    def _append_boundary_entry(
        self,
        page_type: str | None,
        start_file: str | None,
        end_file: str | None,
        date_by_file: dict[str, str],
        report_name_by_file: dict[str, str],
        hearing_boundaries: list[dict[str, str]],
        report_boundaries: list[dict[str, str]],
        minutes_boundaries: list[dict[str, str]],
    ) -> None:
        if not page_type or not start_file or not end_file:
            return
        start_page = _page_label_from_filename(start_file)
        end_page = _page_label_from_filename(end_file)
        if page_type in {"hearing", "hearing_first_page"}:
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
            return
        if page_type in {"minute_order", "minute_order_first_page"}:
            minutes_boundaries.append(
                {
                    "date": date_by_file.get(start_file, ""),
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
        self._raise_if_stop_requested()
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
        payload = _post_json_with_retries(req, timeout=300, error_label="Classifier request failed")
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
        self._raise_if_stop_requested()
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
        payload = _post_json_with_retries(req, timeout=300, error_label="Classifier request failed")
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
