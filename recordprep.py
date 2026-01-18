#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
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
CONFIG_KEY_SELECTED_PDFS = "selected_pdfs"
DEFAULT_CLASSIFIER_PROMPT = (
    "You are labeling a single page of an OCR'd legal transcript. "
    "Return JSON with keys: filename, page_type, date_or_form. "
    "page_type must be one of: report page, hearing page, form page, cover page, index page. "
    "date_or_form should be a date for reports/hearings if present, or the form name if a form. "
    "If unknown, use an empty string."
)


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


class SettingsWindow(Adw.PreferencesWindow):
    def __init__(self, app: Adw.Application, on_saved: Callable[[], None] | None = None) -> None:
        super().__init__(application=app, title="Settings")
        self.set_default_size(640, 480)
        self._on_saved = on_saved

        page = Adw.PreferencesPage(title="LLM Settings")
        credentials_group = Adw.PreferencesGroup(
            title="Classifier Credentials",
            description="OpenAI-compatible endpoint for page classification.",
        )
        self.api_url_row = Adw.EntryRow(title="API URL")
        self.api_url_row.set_hexpand(True)
        credentials_group.add(self.api_url_row)
        self.model_row = Adw.EntryRow(title="Model ID")
        self.model_row.set_hexpand(True)
        credentials_group.add(self.model_row)
        self.api_key_row = self._build_password_row("API Key")
        credentials_group.add(self.api_key_row)
        page.add(credentials_group)

        prompt_group = Adw.PreferencesGroup(title="Classifier Prompt")
        prompt_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        prompt_box.set_margin_top(6)
        prompt_box.set_margin_bottom(6)
        prompt_box.set_margin_start(12)
        prompt_box.set_margin_end(12)
        prompt_label = Gtk.Label(label="Prompt", xalign=0)
        prompt_label.add_css_class("dim-label")
        prompt_box.append(prompt_label)
        prompt_scroller = Gtk.ScrolledWindow()
        prompt_scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        prompt_scroller.set_min_content_height(160)
        self.prompt_buffer = Gtk.TextBuffer()
        prompt_view = Gtk.TextView.new_with_buffer(self.prompt_buffer)
        prompt_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        prompt_view.set_monospace(True)
        prompt_view.set_top_margin(8)
        prompt_view.set_bottom_margin(8)
        prompt_view.set_left_margin(8)
        prompt_view.set_right_margin(8)
        prompt_scroller.set_child(prompt_view)
        prompt_box.append(prompt_scroller)
        prompt_row = Adw.PreferencesRow()
        prompt_row.set_child(prompt_box)
        prompt_group.add(prompt_row)
        page.add(prompt_group)

        actions_group = Adw.PreferencesGroup()
        actions_row = Adw.ActionRow()
        actions_row.set_activatable(False)
        save_btn = Gtk.Button(label="Save Settings")
        save_btn.add_css_class("suggested-action")
        save_btn.connect("clicked", self._on_save_clicked)
        actions_row.add_suffix(save_btn)
        actions_group.add(actions_row)
        page.add(actions_group)

        self._load_settings()
        self.add(page)

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

    def _load_settings(self) -> None:
        settings = load_classifier_settings()
        self.api_url_row.set_text(settings["api_url"])
        self.model_row.set_text(settings["model_id"])
        self.api_key_row.set_text(settings["api_key"])
        self.prompt_buffer.set_text(settings["prompt"])

    def _prompt_text(self) -> str:
        start, end = self.prompt_buffer.get_bounds()
        return self.prompt_buffer.get_text(start, end, True)

    def _on_save_clicked(self, _button: Gtk.Button) -> None:
        save_classifier_settings(
            self.api_url_row.get_text().strip(),
            self.model_row.get_text().strip(),
            self.api_key_row.get_text().strip(),
            self._prompt_text().strip(),
        )
        if self._on_saved:
            self._on_saved()
        self.close()


class RecordPrepWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application) -> None:
        super().__init__(application=app, title=APPLICATION_NAME)
        self.set_default_size(900, 600)

        self.selected_pdfs: list[Path] = []

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

        self.step_two_row = Adw.ActionRow(title="Create classifier file")
        self.step_two_row.set_activatable(True)
        self.step_two_row.connect("activated", self.on_step_two_clicked)
        listbox.append(self.step_two_row)

        self._setup_menu(app)
        self._load_selected_pdfs()

    def _setup_menu(self, app: Adw.Application) -> None:
        menu = Gio.Menu()
        menu.append("Settings", "app.settings")
        self.menu_button.set_menu_model(menu)

        action = Gio.SimpleAction.new("settings", None)
        action.connect("activate", self.on_settings)
        app.add_action(action)

    def on_settings(self, _action: Gio.SimpleAction, _param: object) -> None:
        settings = SettingsWindow(self.get_application(), on_saved=self._on_settings_saved)
        settings.present()

    def _on_settings_saved(self) -> None:
        self.show_toast("Settings saved.")

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
            jsonl_path = root_dir / "classifier.jsonl"
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
        return {
            "filename": str(parsed.get("filename") or filename),
            "page_type": str(parsed.get("page_type") or ""),
            "date_or_form": str(parsed.get("date_or_form") or ""),
        }

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
