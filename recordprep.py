#!/usr/bin/env python3
from __future__ import annotations

import re
import threading
from pathlib import Path

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
    def __init__(self, app: Adw.Application) -> None:
        super().__init__(application=app, title="Settings")
        self.set_default_size(640, 480)

        page = Adw.PreferencesPage(title="LLM Settings")
        group = Adw.PreferencesGroup(
            title="Pipeline Prompts",
            description="Prompt templates and credentials will be configured here.",
        )
        group.add(Adw.ActionRow(title="Coming soon"))
        page.add(group)
        self.add(page)


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

        listbox = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        listbox.add_css_class("boxed-list")
        content.append(listbox)

        self.step_one_row = Adw.ActionRow(title="Create text/image files")
        self.step_one_row.set_activatable(True)
        self.step_one_row.connect("activated", self.on_step_one_clicked)
        listbox.append(self.step_one_row)

        self._setup_menu(app)

    def _setup_menu(self, app: Adw.Application) -> None:
        menu = Gio.Menu()
        menu.append("Settings", "app.settings")
        self.menu_button.set_menu_model(menu)

        action = Gio.SimpleAction.new("settings", None)
        action.connect("activate", self.on_settings)
        app.add_action(action)

    def on_settings(self, _action: Gio.SimpleAction, _param: object) -> None:
        settings = SettingsWindow(self.get_application())
        settings.present()

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
        label = (
            self.selected_pdfs[0].name
            if len(self.selected_pdfs) == 1
            else f"{len(self.selected_pdfs)} PDFs selected"
        )
        self.show_toast(f"Selected: {label}")

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
