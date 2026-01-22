# Record Prep Agent Notes

Project goal:
- GTK4/Libadwaita Python app named "Record Prep" that processes OCR'd legal transcript PDFs into summaries and helper files.

Entry point:
- `recordprep.py` is the application entry and primary UI implementation.

UI expectations:
- Follow patterns from `example_python_GTK4_app/focus.py` when implementing new UI features.
- Each pipeline step is a boxed list row button in the main view.
- Header bar includes a PDF file picker on the left and a hamburger menu on the right.
- Settings opens an Adw.PreferencesWindow for prompt/credential configuration.

Pipeline step 1 (current):
- Create `case_bundle/text_pages` and `case_bundle/image_pages` alongside the input PDF.
- If multiple PDFs are chosen, merge them in alphanumeric order into `case_bundle/temp/merged.pdf` before processing.
- Use `pdftotext` with layout preservation for per-page text files named `0001.txt`, `0002.txt`, etc.
- Render grayscale PNGs at 300 DPI named `0001.png`, `0002.png`, etc. using PyMuPDF.

Dependencies:
- Keep `pyproject.toml` current via `uv add` when adding Python dependencies.
