#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import threading
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
CONFIG_KEY_CLASSIFY_FORMS_API_URL = "classify_form_names_api_url"
CONFIG_KEY_CLASSIFY_FORMS_MODEL_ID = "classify_form_names_model_id"
CONFIG_KEY_CLASSIFY_FORMS_API_KEY = "classify_form_names_api_key"
CONFIG_KEY_CLASSIFY_FORMS_PROMPT = "classify_form_names_prompt"
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


def _format_toc_line(label: str, page: str) -> str:
    if label and page:
        return f"\t{label} {page}"
    if label:
        return f"\t{label}"
    if page:
        return f"\t{page}"
    return "\t"


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
        basic_widgets = self._prompt_editors.get("basic")
        dates_widgets = self._prompt_editors.get("dates")
        report_widgets = self._prompt_editors.get("report-names")
        form_widgets = self._prompt_editors.get("form-names")
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

        self.file_button = Gtk.Button(label="Choose PDF(s)")
        self.file_button.connect("clicked", self.on_choose_pdf)
        header_bar.pack_start(self.file_button)

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

        self.step_one_row = Adw.ActionRow(title="Create text/image files")
        self.step_one_row.set_activatable(True)
        self.step_one_row.connect("activated", self.on_step_one_clicked)
        listbox.append(self.step_one_row)

        self.step_two_row = Adw.ActionRow(title="Create classify basic file")
        self.step_two_row.set_activatable(True)
        self.step_two_row.connect("activated", self.on_step_two_clicked)
        listbox.append(self.step_two_row)

        self.step_three_row = Adw.ActionRow(title="Create classify dates file")
        self.step_three_row.set_activatable(True)
        self.step_three_row.connect("activated", self.on_step_three_clicked)
        listbox.append(self.step_three_row)

        self.step_four_row = Adw.ActionRow(title="Create classify report names file")
        self.step_four_row.set_activatable(True)
        self.step_four_row.connect("activated", self.on_step_four_clicked)
        listbox.append(self.step_four_row)

        self.step_five_row = Adw.ActionRow(title="Create classify form names file")
        self.step_five_row.set_activatable(True)
        self.step_five_row.connect("activated", self.on_step_five_clicked)
        listbox.append(self.step_five_row)

        self.step_six_row = Adw.ActionRow(title="Derive TOC file")
        self.step_six_row.set_activatable(True)
        self.step_six_row.connect("activated", self.on_step_six_clicked)
        listbox.append(self.step_six_row)

        self.step_seven_row = Adw.ActionRow(title="Derive hearing/report Boundaries")
        self.step_seven_row.set_activatable(True)
        self.step_seven_row.connect("activated", self.on_step_seven_clicked)
        listbox.append(self.step_seven_row)

        self._setup_menu(app)
        self._load_selected_pdfs()

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

    def on_step_one_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self.step_one_row.set_sensitive(False)
        threading.Thread(target=self._run_step_one, daemon=True).start()

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

    def on_step_two_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self.step_two_row.set_sensitive(False)
        threading.Thread(target=self._run_step_two, daemon=True).start()

    def on_step_three_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self.step_three_row.set_sensitive(False)
        threading.Thread(target=self._run_step_three, daemon=True).start()

    def on_step_four_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self.step_four_row.set_sensitive(False)
        threading.Thread(target=self._run_step_four, daemon=True).start()

    def on_step_five_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self.step_five_row.set_sensitive(False)
        threading.Thread(target=self._run_step_five, daemon=True).start()

    def on_step_six_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self.step_six_row.set_sensitive(False)
        threading.Thread(target=self._run_step_six, daemon=True).start()

    def on_step_seven_clicked(self, _row: Adw.ActionRow) -> None:
        if not self.selected_pdfs:
            self.show_toast("Choose PDF files first.")
            return
        self.step_seven_row.set_sensitive(False)
        threading.Thread(target=self._run_step_seven, daemon=True).start()

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
