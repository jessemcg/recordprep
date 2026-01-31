# Record Prep Agent Notes

Project goal:
- GTK4/Libadwaita Python app named "Record Prep" that processes OCR'd legal transcript PDFs into summaries and helper files.

Entry point:
- `recordprep.py` is the application entry and primary UI implementation.

UI expectations:
- Follow patterns from `example_python_GTK4_app/focus.py` when implementing new UI features.
- Header bar: case bundle picker + PDF picker on the left, status spinner/label in the center, hamburger menu on the right.
- Main view: "Run all", "Stop", "Resume", and "Edit TOC" buttons plus a boxed list of pipeline step rows.
- Settings: custom `SettingsWindow` (Adw.ApplicationWindow) with a navigation list and prompt editor stack; Save Settings triggers `app.save-settings`.

Pipeline steps (current):
- Create files: create `case_bundle/text_pages` and `case_bundle/image_pages` next to the PDFs. If multiple PDFs are chosen, merge them (natural sort order) into `case_bundle/temp/merged.pdf` first.
- Strip characters: remove non-printing characters from extracted text files.
- Infer case: infer the case name from the first pages and write `case_bundle/case_name.txt`.
- Classification basic: create `classification/basic.jsonl` for every page.
- Correct basic classification: write `classification/basic_corrected.jsonl`.
- Advanced classification: annotate hearing last pages and minute/form first pages in `basic_corrected_advanced.jsonl`.
- Correct advanced classification: fix consecutive last-page markers in `basic_corrected_advanced_corrected.jsonl`.
- Classification dates: add dates for hearing and minute order first pages in `basic_corrected_advanced_corrected_dates.jsonl`.
- Classification names: add report/form names in `basic_corrected_advanced_corrected_dates_names.jsonl`.
- Build TOC: generate `artifacts/toc.txt`.
- Correct TOC: remove duplicate minute order dates in the TOC.
- Find boundaries: write `artifacts/hearing_boundaries.json`, `artifacts/report_boundaries.json`, and `artifacts/minutes_boundaries.json`.
- Create raw: write `artifacts/raw_hearings.txt` and `artifacts/raw_reports.txt`.
- Create optimized: write `artifacts/optimized_hearings.txt` and `artifacts/optimized_reports.txt`.
- Create summaries: write case-named summary files in `summaries/` (fallback to `summarized_*.txt` when needed).
- Case overview: write `rag/case_overview.txt`.
- Create RAG index: build `rag/vector_database` with VoyageAI + Chroma.

Step 1 implementation details:
- Use `pdftotext` with `physical=True` to create per-page text files named `0001.txt`, `0002.txt`, etc.
- Render grayscale PNGs at 300 DPI named `0001.png`, `0002.png`, etc. using PyMuPDF.

Dependencies:
- Keep `pyproject.toml` current via `uv add` when adding Python dependencies.
