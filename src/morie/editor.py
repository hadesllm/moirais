"""Built-in scientific computing editor for MORIE.

A nano-like text editor built on Textual with line numbers, syntax awareness,
and the ability to run the file directly (ctrl+r).

Usage:
    morie edit myfile.py           # open or create
    morie edit analysis.R --run    # enable ctrl+r to run after save
    morie exec co test.py          # create in cofs/ and open
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def launch_editor(filepath: str, run_on_save: bool = False, lang_hint: str | None = None) -> int:
    try:
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Vertical
        from textual.widgets import Footer, Header, Static, TextArea
    except ImportError:
        print("Textual not available. Install with: pip install textual")
        print(f"Falling back to nano for {filepath}")
        return subprocess.call(["nano", filepath])

    path = Path(filepath).resolve()
    ext = path.suffix.lower()
    if lang_hint is None:
        lang_map = {
            ".py": "python",
            ".r": "r",
            ".sh": "bash",
            ".js": "javascript",
            ".rs": "rust",
            ".c": "c",
            ".cpp": "cpp",
            ".sql": "sql",
            ".go": "go",
            ".ml": "ocaml",
            ".mli": "ocaml",
            ".lua": "lua",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".tex": "latex",
            ".md": "markdown",
            ".yml": "yaml",
            ".yaml": "yaml",
            ".toml": "toml",
            ".json": "json",
            ".csv": "css",
        }
        lang_hint = lang_map.get(ext, "python")

    textual_lang = lang_hint if lang_hint in ("python", "markdown", "json", "yaml", "toml", "sql", "css") else None

    class MORIEEditor(App):
        CSS = """
        #status { height: 1; background: $accent; color: $text; padding: 0 1; }
        #output { height: auto; max-height: 10; background: $surface; padding: 0 1; display: none; }
        TextArea { height: 1fr; }
        """
        BINDINGS = [
            Binding("ctrl+s", "save", "Save"),
            Binding("ctrl+r", "run_file", "Save+Run"),
            Binding("ctrl+w", "save_as", "Save As"),
            Binding("ctrl+b", "go_back", "Back"),
            Binding("ctrl+q", "quit_editor", "Save+Quit"),
        ]

        def __init__(self):
            super().__init__()
            self._modified = False
            self._filepath = path

        def _status_text(self, extra: str = "") -> str:
            mod = " [modified]" if self._modified else ""
            ex = f" {extra}" if extra else ""
            return f" {self._filepath.name}{mod}{ex} | {lang_hint} | ^S save ^R run ^W save-as ^B back ^Q quit"

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Vertical():
                content = path.read_text() if path.exists() else ""
                ta = TextArea(content, id="editor", show_line_numbers=True)
                if textual_lang:
                    ta.language = textual_lang
                yield ta
                yield Static("", id="output")
                yield Static(self._status_text(), id="status")
            yield Footer()

        def on_text_area_changed(self, event) -> None:
            self._modified = True
            self.query_one("#status", Static).update(self._status_text())

        def action_save(self) -> None:
            content = self.query_one("#editor", TextArea).text
            self._filepath.parent.mkdir(parents=True, exist_ok=True)
            self._filepath.write_text(content)
            self._modified = False
            self.query_one("#status", Static).update(self._status_text("[saved]"))

        def action_save_as(self) -> None:
            self.action_save()
            out = self.query_one("#output", Static)
            out.styles.display = "block"
            out.update(f"File saved to: {self._filepath}\nTo save elsewhere: cp {self._filepath} <dest>")

        def action_run_file(self) -> None:
            self.action_save()
            out_widget = self.query_one("#output", Static)
            out_widget.styles.display = "block"
            out_widget.update("Running...")

            if ext in (".py",) or lang_hint == "python":
                cmd = [sys.executable, str(self._filepath)]
            elif ext in (".r", ".R") or lang_hint == "r":
                cmd = ["Rscript", str(self._filepath)]
            elif ext in (".sh", ".bash") or lang_hint == "shell":
                cmd = ["bash", str(self._filepath)]
            else:
                cmd = [sys.executable, str(self._filepath)]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                output = result.stdout[-500:] if result.stdout else ""
                if result.stderr:
                    output += "\n" + result.stderr[-300:]
                out_widget.update(output.strip() or "(no output)")
            except subprocess.TimeoutExpired:
                out_widget.update("Timeout (60s)")
            except Exception as e:
                out_widget.update(f"Error: {e}")

        def action_go_back(self) -> None:
            self.exit(0)

        def action_quit_editor(self) -> None:
            if self._modified:
                self.action_save()
            self.exit(0)

    app = MORIEEditor()
    app.run()
    return 0
