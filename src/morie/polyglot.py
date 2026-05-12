"""
Headless polyglot REPL engine -- multi-language execution with variable bridging.

Supports: Python, R, Shell (bash/zsh), Julia, SQL (SQLite/PostgreSQL), Node.js,
          Go, Rust, C, C++, OCaml, Lua, TypeScript, LaTeX
Variable bridging: automatic bidirectional across all active languages.
"""

from __future__ import annotations

import code as _code
import io
import os
import re
import subprocess
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field

_R_PATTERNS = {
    "<-",
    "%>%",
    "%in%",
    "|>",
    "library(",
    "require(",
    "data.frame(",
    "read.csv(",
    "install.packages(",
    "dplyr::",
    "tidyr::",
    "ggplot::",
    "purrr::",
    "ggplot(",
    "aes(",
    "tibble(",
    "lm(",
    "glm(",
    "t.test(",
    "chisq.test(",
    "c(",
    "seq(",
    "rep(",
    "paste0(",
    "nrow(",
    "ncol(",
    "summary(",
    "str(",
    "head(",
    "tail(",
}

_SHELL_COMMANDS = frozenset(
    {
        "ls",
        "cd",
        "pwd",
        "cat",
        "grep",
        "find",
        "echo",
        "mkdir",
        "rm",
        "cp",
        "mv",
        "chmod",
        "chown",
        "git",
        "docker",
        "brew",
        "apt",
        "pip",
        "npm",
        "curl",
        "wget",
        "ssh",
        "scp",
        "tar",
        "zip",
        "which",
        "whoami",
        "date",
        "env",
        "export",
        "source",
        "man",
        "head",
        "tail",
        "wc",
        "sort",
        "uniq",
        "sed",
        "awk",
        "cut",
        "tr",
        "xargs",
        "tee",
        "touch",
        "ollama",
        "morie",
        "python3",
        "Rscript",
        "node",
        "julia",
    }
)

_JULIA_PATTERNS = {
    "using ",
    "println(",
    "typeof(",
    "eltype(",
    "::Int64",
    "::Float64",
    "::String",
    "::Vector{",
    "::Matrix{",
    "@.",
    "@time ",
    "@show ",
    "@assert ",
}

_SQL_START_PATTERNS = {
    "SELECT ",
    "INSERT ",
    "UPDATE ",
    "DELETE ",
    "CREATE ",
    "DROP ",
    "ALTER ",
    "PRAGMA ",
    ".tables",
    ".schema",
    "EXPLAIN ",
}

_SQL_PATTERNS = _SQL_START_PATTERNS

_NODE_PATTERNS = {
    "console.log(",
    "require(",
    "const ",
    "let ",
    "var ",
    "async ",
    "await ",
    "=>(",
    "=> {",
    "process.",
    "Buffer.",
    "Promise.",
    "JSON.parse(",
    "JSON.stringify(",
}

_GO_PATTERNS = {
    "package main",
    "fmt.Print",
    "fmt.Scan",
    "func main()",
    "go func",
    "chan ",
    "defer ",
    ":= ",
    "import (",
}

_RUST_PATTERNS = {
    "fn ",
    "let mut ",
    "println!(",
    "eprintln!(",
    "impl ",
    "pub fn ",
    "struct ",
    "enum ",
    "use std::",
    "match ",
    "unwrap()",
    "expect(",
    "Vec<",
    "String::",
    "Option<",
    "Result<",
}

_C_PATTERNS = {
    "#include <",
    "printf(",
    "scanf(",
    "int main(",
    "void ",
    "malloc(",
    "free(",
    "sizeof(",
    "NULL",
    "typedef ",
}

_CPP_PATTERNS = {
    "std::",
    "cout <<",
    "cin >>",
    "#include <iostream>",
    "vector<",
    "template<",
    "unique_ptr<",
    "shared_ptr<",
    "namespace ",
    "nullptr",
}

_OCAML_PATTERNS = {
    "Printf.printf",
    "List.map",
    "Array.make",
    "List.iter",
    "List.fold_left",
    ";;",
    "let () =",
    "fun x ->",
    "fun _ ->",
    "Hashtbl.",
    "Sys.argv",
    "Printexc.",
}

_LUA_PATTERNS = {
    "local ",
    "io.read(",
    "io.write(",
    "table.insert(",
    "table.remove(",
    "string.format(",
    "elseif",
    "pairs(",
    "ipairs(",
    "tonumber(",
    "tostring(",
}

_TYPESCRIPT_PATTERNS = {
    "interface ",
    ": string",
    ": number",
    ": boolean",
    "export ",
    "import {",
    "type ",
    "as ",
    "extends ",
    "implements ",
    "<T>",
    "Promise<",
    "readonly ",
}

_LATEX_PATTERNS = {
    "\\documentclass",
    "\\begin{",
    "\\end{",
    "\\section{",
    "\\subsection{",
    "\\usepackage{",
    "\\textbf{",
    "\\textit{",
    "\\frac{",
    "\\alpha",
    "\\beta",
    "\\sum",
    "\\label{",
    "\\ref{",
    "\\cite{",
}

_PSQL_PATTERNS = {
    "\\dt",
    "\\d+",
    "\\l",
    "\\c ",
    "\\conninfo",
    "\\copy",
    "\\timing",
    "\\pset",
    "RETURNING ",
    "ON CONFLICT ",
    "LATERAL ",
}

_PERL_PATTERNS = {
    "my $",
    "my @",
    "my %",
    "chomp(",
    "=~ ",
    "=~ s/",
    "use strict",
    "use warnings",
    "qw(",
    "die ",
}

_RUBY_PATTERNS = {
    "puts ",
    "require_relative",
    ".each do",
    ".map {",
    "attr_accessor",
    "elsif",
    "|x|",
    "Gem::",
    "ARGV[",
}

_PHP_PATTERNS = {
    "<?php",
    "$_GET",
    "$_POST",
    "$_SERVER",
    "var_dump(",
    "strlen(",
    "isset(",
    'echo "',
}

_JAVA_PATTERNS = {
    "public class ",
    "public static void main",
    "System.out.print",
    "import java.",
    "throws ",
    "ArrayList<",
    "HashMap<",
}

_KOTLIN_PATTERNS = {
    "fun main(",
    "listOf(",
    "mapOf(",
    "data class ",
    "companion object",
    "?.let",
    "when (",
}

_SCALA_PATTERNS = {
    "def main(args:",
    "case class ",
    "import scala.",
    "Seq(",
    "match {",
    "extends App",
}

_SWIFT_PATTERNS = {
    "import Foundation",
    "import UIKit",
    "import SwiftUI",
    "guard let ",
    "if let ",
    "@objc",
    "@State",
    "@Published",
    "protocol ",
    "extension ",
}

_HASKELL_PATTERNS = {
    "import qualified",
    "putStrLn",
    "deriving",
    "newtype ",
    "main :: IO ()",
    "Prelude.",
    "Data.List",
}

_ELIXIR_PATTERNS = {
    "defmodule ",
    "defp ",
    "IO.puts",
    "Enum.",
    "%{",
    "iex>",
    "GenServer",
    "Supervisor",
}

_ZIG_PATTERNS = {
    "const std = @import",
    "pub fn ",
    "std.debug.print",
    "std.mem.",
    "comptime ",
    "errdefer ",
    "try ",
    "catch ",
}

_NIM_PATTERNS = {
    "proc ",
    "echo ",
    "discard ",
    "nimble ",
    "when defined(",
    "import strutils",
    "import os",
    "import sequtils",
}

_D_PATTERNS = {
    "import std.stdio",
    "writeln(",
    "writef(",
    "immutable ",
    "unittest {",
}

_FORTRAN_PATTERNS = {
    "implicit none",
    "write(*,",
    "read(*,",
    "subroutine ",
    "end program",
    "end do",
    "end subroutine",
}

_OCTAVE_PATTERNS = {
    "disp(",
    "fprintf(",
    "sprintf(",
    "zeros(",
    "ones(",
    "linspace(",
    "plot(",
    "xlabel(",
    "ylabel(",
    "function ",
    "endfunction",
    "pkg load",
}

_PROLOG_PATTERNS = {
    ":-",
    "?-",
    "assert(",
    "retract(",
    "findall(",
    "write(",
    "nl.",
    "is ",
    "not(",
}

_SCHEME_PATTERNS = {
    "(define ",
    "(lambda ",
    "(let ",
    "(display ",
    "(newline)",
    "(begin ",
    "(cond ",
    "(if ",
    "(car ",
    "(cdr ",
    "#lang racket",
    "(require ",
}

_CLOJURE_PATTERNS = {
    "(ns ",
    "(defn ",
    "(def ",
    "(println ",
    "(str ",
    "(map ",
    "(filter ",
    "(reduce ",
    "(require ",
    "clojure.",
}

_DART_PATTERNS = {
    "import 'dart:",
    "Widget build(",
    "StatelessWidget",
    "StatefulWidget",
    "BuildContext",
    "Future<",
    "Stream<",
}

_POWERSHELL_PATTERNS = {
    "Write-Host",
    "Get-",
    "Set-",
    "$PSVersionTable",
    "param(",
    "ForEach-Object",
    "Where-Object",
    "[string]",
    "[int]",
    "| Select",
    "-eq ",
    "-ne ",
    "-like ",
}

_AWK_PATTERNS = {
    "BEGIN {",
    "END {",
    "NR",
    "NF",
    "$0",
    "$1",
    "$2",
    "print $",
    "printf ",
    "FS=",
    "OFS=",
    "/pattern/",
}

_RMD_PATTERNS = {
    "```{r",
    "```{python",
    "knitr::",
    "rmarkdown::",
    "---\ntitle:",
    "output:",
}

LANGUAGES = (
    "python",
    "r",
    "shell",
    "julia",
    "sql",
    "node",
    "go",
    "rust",
    "c",
    "cpp",
    "ocaml",
    "lua",
    "typescript",
    "latex",
    "psql",
    "perl",
    "ruby",
    "php",
    "java",
    "kotlin",
    "scala",
    "swift",
    "haskell",
    "elixir",
    "zig",
    "nim",
    "d",
    "fortran",
    "octave",
    "prolog",
    "scheme",
    "clojure",
    "dart",
    "powershell",
    "awk",
    "rmd",
)

LABELS = {
    "python": "[P]",
    "r": "[R]",
    "shell": "[S]",
    "julia": "[J]",
    "sql": "[Q]",
    "node": "[N]",
    "go": "[Go]",
    "rust": "[Rs]",
    "c": "[C]",
    "cpp": "[C+]",
    "ocaml": "[ML]",
    "lua": "[Lu]",
    "typescript": "[TS]",
    "latex": "[TX]",
    "psql": "[PG]",
    "perl": "[Pl]",
    "ruby": "[Rb]",
    "php": "[HP]",
    "java": "[Jv]",
    "kotlin": "[Kt]",
    "scala": "[Sc]",
    "swift": "[Sw]",
    "haskell": "[Hs]",
    "elixir": "[Ex]",
    "zig": "[Zg]",
    "nim": "[Nm]",
    "d": "[D]",
    "fortran": "[F]",
    "octave": "[Oc]",
    "prolog": "[Pr]",
    "scheme": "[Sk]",
    "clojure": "[Cj]",
    "dart": "[Dt]",
    "powershell": "[PS]",
    "awk": "[Aw]",
    "rmd": "[Rm]",
}


def detect_language(code: str, default: str = "python") -> str:
    stripped = code.strip()
    if not stripped:
        return default

    if stripped.startswith("!") or stripped.startswith("$"):
        return "shell"

    _PREFIX_MAP = [
        ("R>", "r"),
        ("r>", "r"),
        ("J>", "julia"),
        ("j>", "julia"),
        ("Q>", "sql"),
        ("q>", "sql"),
        ("N>", "node"),
        ("n>", "node"),
        ("Go>", "go"),
        ("go>", "go"),
        ("Rs>", "rust"),
        ("rs>", "rust"),
        ("C+>", "cpp"),
        ("ML>", "ocaml"),
        ("ml>", "ocaml"),
        ("Lu>", "lua"),
        ("lu>", "lua"),
        ("TS>", "typescript"),
        ("ts>", "typescript"),
        ("TX>", "latex"),
        ("tx>", "latex"),
        ("PG>", "psql"),
        ("pg>", "psql"),
        ("Pl>", "perl"),
        ("pl>", "perl"),
        ("Rb>", "ruby"),
        ("rb>", "ruby"),
        ("HP>", "php"),
        ("hp>", "php"),
        ("Jv>", "java"),
        ("jv>", "java"),
        ("Kt>", "kotlin"),
        ("kt>", "kotlin"),
        ("Sc>", "scala"),
        ("sc>", "scala"),
        ("Sw>", "swift"),
        ("sw>", "swift"),
        ("Hs>", "haskell"),
        ("hs>", "haskell"),
        ("Ex>", "elixir"),
        ("ex>", "elixir"),
        ("Zg>", "zig"),
        ("zg>", "zig"),
        ("Nm>", "nim"),
        ("nm>", "nim"),
        ("D>", "d"),
        ("d>", "d"),
        ("F>", "fortran"),
        ("f>", "fortran"),
        ("Oc>", "octave"),
        ("oc>", "octave"),
        ("Pr>", "prolog"),
        ("pr>", "prolog"),
        ("Sk>", "scheme"),
        ("sk>", "scheme"),
        ("Cj>", "clojure"),
        ("cj>", "clojure"),
        ("Dt>", "dart"),
        ("dt>", "dart"),
        ("PS>", "powershell"),
        ("ps>", "powershell"),
        ("Aw>", "awk"),
        ("aw>", "awk"),
        ("Rm>", "rmd"),
        ("rm>", "rmd"),
        ("C>", "c"),
    ]
    for prefix, lang in _PREFIX_MAP:
        if stripped.startswith(prefix):
            return lang

    upper = stripped.upper()
    for pat in _SQL_START_PATTERNS:
        if upper.startswith(pat):
            return "sql"

    for pat in _PSQL_PATTERNS:
        if pat in code:
            return "psql"

    for pat in _LATEX_PATTERNS:
        if pat in code:
            return "latex"

    for pat in _R_PATTERNS:
        if pat in code:
            if pat == "c(" and re.match(r"^c\s*=", stripped):
                continue
            return "r"

    for pat in _GO_PATTERNS:
        if pat in code:
            return "go"

    for pat in _RUST_PATTERNS:
        if pat in code:
            return "rust"

    for pat in _CPP_PATTERNS:
        if pat in code:
            return "cpp"

    for pat in _C_PATTERNS:
        if pat in code:
            return "c"

    for pat in _OCAML_PATTERNS:
        if pat in code:
            return "ocaml"

    for pat in _JULIA_PATTERNS:
        if pat in code:
            return "julia"

    for pat in _TYPESCRIPT_PATTERNS:
        if pat in code:
            return "typescript"

    for pat in _NODE_PATTERNS:
        if pat in code:
            return "node"

    for pat in _LUA_PATTERNS:
        if pat in code:
            return "lua"

    for pat in _PERL_PATTERNS:
        if pat in code:
            return "perl"

    for pat in _RUBY_PATTERNS:
        if pat in code:
            return "ruby"

    if stripped.startswith("<?php") or any(p in code for p in _PHP_PATTERNS):
        return "php"

    for pat in _JAVA_PATTERNS:
        if pat in code:
            return "java"

    for pat in _KOTLIN_PATTERNS:
        if pat in code:
            return "kotlin"

    for pat in _SCALA_PATTERNS:
        if pat in code:
            return "scala"

    for pat in _SWIFT_PATTERNS:
        if pat in code:
            return "swift"

    for pat in _HASKELL_PATTERNS:
        if pat in code:
            return "haskell"

    for pat in _ELIXIR_PATTERNS:
        if pat in code:
            return "elixir"

    for pat in _ZIG_PATTERNS:
        if pat in code:
            return "zig"

    for pat in _NIM_PATTERNS:
        if pat in code:
            return "nim"

    for pat in _D_PATTERNS:
        if pat in code:
            return "d"

    for pat in _FORTRAN_PATTERNS:
        if pat in code:
            return "fortran"

    for pat in _OCTAVE_PATTERNS:
        if pat in code:
            return "octave"

    for pat in _PROLOG_PATTERNS:
        if pat in code:
            return "prolog"

    for pat in _SCHEME_PATTERNS:
        if pat in code:
            return "scheme"

    for pat in _CLOJURE_PATTERNS:
        if pat in code:
            return "clojure"

    for pat in _DART_PATTERNS:
        if pat in code:
            return "dart"

    for pat in _POWERSHELL_PATTERNS:
        if pat in code:
            return "powershell"

    for pat in _AWK_PATTERNS:
        if pat in code:
            return "awk"

    for pat in _RMD_PATTERNS:
        if pat in code:
            return "rmd"

    first_word = stripped.split()[0] if stripped.split() else ""
    if first_word in _SHELL_COMMANDS:
        return "shell"

    return default


@dataclass
class ExecResult:
    language: str
    stdout: str = ""
    stderr: str = ""
    success: bool = True
    variables: dict = field(default_factory=dict)


class PolyglotEngine:
    """Multi-language REPL engine with variable bridging."""

    def __init__(
        self,
        polyglot: bool = True,
        auto_detect: bool = True,
        output_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.polyglot = polyglot
        self.auto_detect = auto_detect
        self._output = output_fn or (lambda s: print(s))
        self._default_lang = "python"

        self._py_ns: dict = {"__name__": "__morie_repl__", "__builtins__": __builtins__}
        self._py_console = _code.InteractiveConsole(self._py_ns)

        self._r_proc: subprocess.Popen | None = None
        self._julia_proc: subprocess.Popen | None = None
        self._node_proc: subprocess.Popen | None = None
        self._shell = os.environ.get("SHELL", "/bin/bash")

        self._db_conn = None

        self._preload_python()

    def _preload_python(self) -> None:
        try:
            import numpy as np
            import pandas as pd

            self._py_ns["np"] = np
            self._py_ns["pd"] = pd
        except ImportError:
            pass
        try:
            from morie import fn
            from morie.fn._registry import REGISTRY

            self._py_ns["fn"] = fn
            self._py_ns["REGISTRY"] = REGISTRY
            for key, entry in REGISTRY.items():
                try:
                    mod = __import__(f"morie.fn.{key}", fromlist=[entry.func_name])
                    self._py_ns[key] = getattr(mod, entry.func_name)
                except Exception:
                    pass
        except ImportError:
            pass

    def _start_r(self) -> bool:
        if self._r_proc and self._r_proc.poll() is None:
            return True
        try:
            self._r_proc = subprocess.Popen(
                ["R", "--no-echo", "--no-save", "--no-restore"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            return True
        except FileNotFoundError:
            return False

    def _start_julia(self) -> bool:
        if self._julia_proc and self._julia_proc.poll() is None:
            return True
        try:
            self._julia_proc = subprocess.Popen(
                ["julia", "--startup-file=no", "-q"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            return True
        except FileNotFoundError:
            return False

    def _start_node(self) -> bool:
        if self._node_proc and self._node_proc.poll() is None:
            return True
        try:
            self._node_proc = subprocess.Popen(
                ["node", "-i"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env={**os.environ, "NODE_NO_READLINE": "1"},
            )
            return True
        except FileNotFoundError:
            return False

    def _start_sql(self) -> bool:
        if self._db_conn is not None:
            return True
        try:
            import sqlite3

            db_path = os.environ.get("MORIE_DB", ":memory:")
            if db_path == ":memory:":
                try:
                    from morie.data import _builtin_db_path

                    candidate = _builtin_db_path()
                    if os.path.exists(candidate):
                        db_path = candidate
                except Exception:
                    pass
            self._db_conn = sqlite3.connect(db_path)
            return True
        except Exception:
            return False

    def strip_prefix(self, code: str) -> tuple[str, str | None]:
        stripped = code.strip()
        for prefix, lang in [
            ("Go>", "go"),
            ("go>", "go"),
            ("Rs>", "rust"),
            ("rs>", "rust"),
            ("C+>", "cpp"),
            ("ML>", "ocaml"),
            ("ml>", "ocaml"),
            ("Lu>", "lua"),
            ("lu>", "lua"),
            ("TS>", "typescript"),
            ("ts>", "typescript"),
            ("TX>", "latex"),
            ("tx>", "latex"),
            ("PG>", "psql"),
            ("pg>", "psql"),
            ("Pl>", "perl"),
            ("pl>", "perl"),
            ("Rb>", "ruby"),
            ("rb>", "ruby"),
            ("HP>", "php"),
            ("hp>", "php"),
            ("Jv>", "java"),
            ("jv>", "java"),
            ("Kt>", "kotlin"),
            ("kt>", "kotlin"),
            ("Sc>", "scala"),
            ("sc>", "scala"),
            ("Sw>", "swift"),
            ("sw>", "swift"),
            ("Hs>", "haskell"),
            ("hs>", "haskell"),
            ("Ex>", "elixir"),
            ("ex>", "elixir"),
            ("Zg>", "zig"),
            ("zg>", "zig"),
            ("Nm>", "nim"),
            ("nm>", "nim"),
            ("D>", "d"),
            ("d>", "d"),
            ("F>", "fortran"),
            ("f>", "fortran"),
            ("Oc>", "octave"),
            ("oc>", "octave"),
            ("Pr>", "prolog"),
            ("pr>", "prolog"),
            ("Sk>", "scheme"),
            ("sk>", "scheme"),
            ("Cj>", "clojure"),
            ("cj>", "clojure"),
            ("Dt>", "dart"),
            ("dt>", "dart"),
            ("PS>", "powershell"),
            ("ps>", "powershell"),
            ("Aw>", "awk"),
            ("aw>", "awk"),
            ("Rm>", "rmd"),
            ("rm>", "rmd"),
            ("R>", "r"),
            ("r>", "r"),
            ("J>", "julia"),
            ("j>", "julia"),
            ("Q>", "sql"),
            ("q>", "sql"),
            ("N>", "node"),
            ("n>", "node"),
            ("C>", "c"),
            ("!", "shell"),
            ("$", "shell"),
        ]:
            if stripped.startswith(prefix):
                return stripped[len(prefix) :].strip(), lang
        return stripped, None

    def execute(self, code: str) -> ExecResult:
        actual_code, forced_lang = self.strip_prefix(code)
        lang = forced_lang or (detect_language(code, self._default_lang) if self.auto_detect else self._default_lang)

        _dispatch = {
            "python": self._exec_python,
            "r": self._exec_r,
            "shell": self._exec_shell,
            "julia": self._exec_julia,
            "sql": self._exec_sql,
            "node": self._exec_node,
            "go": self._exec_go,
            "rust": self._exec_rust,
            "c": self._exec_c,
            "cpp": self._exec_cpp,
            "ocaml": self._exec_ocaml,
            "lua": self._exec_lua,
            "typescript": self._exec_typescript,
            "latex": self._exec_latex,
            "psql": self._exec_psql,
            "perl": self._exec_perl,
            "ruby": self._exec_ruby,
            "php": self._exec_php,
            "java": self._exec_java,
            "kotlin": self._exec_kotlin,
            "scala": self._exec_scala,
            "swift": self._exec_swift,
            "haskell": self._exec_haskell,
            "elixir": self._exec_elixir,
            "zig": self._exec_zig,
            "nim": self._exec_nim,
            "d": self._exec_d,
            "fortran": self._exec_fortran,
            "octave": self._exec_octave,
            "prolog": self._exec_prolog,
            "scheme": self._exec_scheme,
            "clojure": self._exec_clojure,
            "dart": self._exec_dart,
            "powershell": self._exec_powershell,
            "awk": self._exec_awk,
            "rmd": self._exec_rmd,
        }
        handler = _dispatch.get(lang)
        if handler:
            result = handler(actual_code)
        else:
            result = ExecResult(language=lang, stderr=f"Unknown language: {lang}", success=False)

        if self.polyglot and result.success and result.variables:
            self._bridge_variables(lang, result.variables)

        return result

    def _exec_python(self, code: str) -> ExecResult:
        stdout_cap = io.StringIO()
        stderr_cap = io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        variables = {}
        try:
            sys.stdout = stdout_cap
            sys.stderr = stderr_cap
            if "\n" in code:
                compiled = compile(code, "<repl>", "exec")
                exec(compiled, self._py_ns)
            else:
                self._py_console.push(code)
            sys.stdout = old_out
            sys.stderr = old_err

            for m in re.finditer(r"^(\w+)\s*(?<![!=<>+\-*/])=(?!=)", code, re.MULTILINE):
                name = m.group(1)
                if name in self._py_ns:
                    variables[name] = self._py_ns[name]

            err_text = stderr_cap.getvalue()
            return ExecResult(
                language="python",
                stdout=stdout_cap.getvalue(),
                stderr=err_text,
                success="Error" not in err_text and "Traceback" not in err_text,
                variables=variables,
            )
        except Exception:
            sys.stdout = old_out
            sys.stderr = old_err
            tb = traceback.format_exc().strip().splitlines()
            return ExecResult(language="python", stderr=tb[-1] if tb else "Error", success=False)
        finally:
            sys.stdout = old_out
            sys.stderr = old_err

    def _exec_r(self, code: str) -> ExecResult:
        if not self._start_r():
            return self._exec_r_oneshot(code)

        if self.polyglot:
            for name, val in self._py_ns.items():
                if name.startswith("_") or not isinstance(val, (int, float, str, bool)):
                    continue
                self._inject_r_var(name, val)

        sentinel = f"__MORIE_{id(code)}__"
        try:
            self._r_proc.stdin.write(f"{code}\ncat('{sentinel}\\n')\n")
            self._r_proc.stdin.flush()
            lines = []
            while True:
                raw = self._r_proc.stdout.readline()
                if not raw:
                    break
                if sentinel in raw:
                    break
                lines.append(raw.rstrip("\n"))

            variables = {}
            for m in re.finditer(r"(\w+)\s*<-", code):
                name = m.group(1)
                variables[name] = self._get_r_value(name)

            return ExecResult(language="r", stdout="\n".join(lines), variables=variables)
        except Exception as e:
            return ExecResult(language="r", stderr=str(e), success=False)

    def _exec_r_oneshot(self, code: str) -> ExecResult:
        try:
            result = subprocess.run(
                ["Rscript", "-e", code],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return ExecResult(
                language="r",
                stdout=result.stdout,
                stderr=result.stderr,
                success=result.returncode == 0,
            )
        except FileNotFoundError:
            return ExecResult(language="r", stderr="R not found", success=False)
        except subprocess.TimeoutExpired:
            return ExecResult(language="r", stderr="Timeout (30s)", success=False)

    def _get_r_value(self, name: str) -> object:
        if not self._r_proc or self._r_proc.poll() is not None:
            return None
        sentinel = f"__VAL_{id(name)}__"
        try:
            cmd = (
                f"tryCatch({{"
                f"  .v <- {name};"
                f"  if(is.data.frame(.v)) cat('__DF__\\n')"
                f"  else if(length(.v)==1) cat(as.character(.v), '\\n')"
                f"  else cat(paste0(.v, collapse=','), '\\n')"
                f"}}, error=function(e) cat('__ERR__\\n'))\n"
                f"cat('{sentinel}\\n')\n"
            )
            self._r_proc.stdin.write(cmd)
            self._r_proc.stdin.flush()
            lines = []
            while True:
                raw = self._r_proc.stdout.readline()
                if not raw or sentinel in raw:
                    break
                lines.append(raw.rstrip("\n"))
            val = " ".join(lines).strip()
            if val == "__ERR__" or not val:
                return None
            if val == "__DF__":
                return "DataFrame"
            if val in ("TRUE", "FALSE"):
                return val == "TRUE"
            try:
                return int(val)
            except ValueError:
                pass
            try:
                return float(val)
            except ValueError:
                pass
            if "," in val:
                parts = val.split(",")
                try:
                    return [float(p) if "." in p else int(p) for p in parts]
                except ValueError:
                    return parts
            return val
        except Exception:
            return None

    def _exec_shell(self, code: str) -> ExecResult:
        try:
            result = subprocess.run(
                [self._shell, "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, **{k: str(v) for k, v in self._py_ns.items() if isinstance(v, (str, int, float))}},
            )
            variables = {}
            for m in re.finditer(r'(\w+)=(["\']?)(.+?)\2(?:\s|$)', code):
                name, _, val = m.groups()
                try:
                    variables[name] = int(val)
                except ValueError:
                    try:
                        variables[name] = float(val)
                    except ValueError:
                        variables[name] = val
            return ExecResult(
                language="shell",
                stdout=result.stdout,
                stderr=result.stderr,
                success=result.returncode == 0,
                variables=variables,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(language="shell", stderr="Timeout (30s)", success=False)

    def _exec_julia(self, code: str) -> ExecResult:
        if not self._start_julia():
            return self._exec_julia_oneshot(code)

        sentinel = f"__MORIE_{id(code)}__"
        try:
            self._julia_proc.stdin.write(f'{code}\nprintln("{sentinel}")\n')
            self._julia_proc.stdin.flush()
            lines = []
            while True:
                raw = self._julia_proc.stdout.readline()
                if not raw or sentinel in raw:
                    break
                lines.append(raw.rstrip("\n"))

            variables = {}
            for m in re.finditer(r"(\w+)\s*=", code):
                variables[m.group(1)] = None

            return ExecResult(language="julia", stdout="\n".join(lines), variables=variables)
        except Exception as e:
            return ExecResult(language="julia", stderr=str(e), success=False)

    def _exec_julia_oneshot(self, code: str) -> ExecResult:
        try:
            result = subprocess.run(
                ["julia", "-e", code],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return ExecResult(
                language="julia",
                stdout=result.stdout,
                stderr=result.stderr,
                success=result.returncode == 0,
            )
        except FileNotFoundError:
            return ExecResult(language="julia", stderr="Julia not found", success=False)
        except subprocess.TimeoutExpired:
            return ExecResult(language="julia", stderr="Timeout (30s)", success=False)

    def _exec_sql(self, code: str) -> ExecResult:
        if not self._start_sql():
            return ExecResult(language="sql", stderr="SQLite unavailable", success=False)
        try:
            cursor = self._db_conn.execute(code)
            if cursor.description:
                cols = [d[0] for d in cursor.description]
                rows = cursor.fetchall()
                header = " | ".join(cols)
                sep = "-+-".join("-" * len(c) for c in cols)
                body = "\n".join(" | ".join(str(v) for v in row) for row in rows[:100])
                stdout = f"{header}\n{sep}\n{body}" if rows else f"{header}\n(0 rows)"
                try:
                    import pandas as pd

                    df = pd.DataFrame(rows, columns=cols)
                    self._py_ns["_last_query"] = df
                except ImportError:
                    pass
                return ExecResult(language="sql", stdout=stdout, variables={"_last_query": None})
            else:
                self._db_conn.commit()
                return ExecResult(language="sql", stdout=f"OK ({cursor.rowcount} rows affected)")
        except Exception as e:
            return ExecResult(language="sql", stderr=str(e), success=False)

    def _exec_node(self, code: str) -> ExecResult:
        try:
            result = subprocess.run(
                ["node", "-e", code],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return ExecResult(
                language="node",
                stdout=result.stdout,
                stderr=result.stderr,
                success=result.returncode == 0,
            )
        except FileNotFoundError:
            return ExecResult(language="node", stderr="Node.js not found", success=False)
        except subprocess.TimeoutExpired:
            return ExecResult(language="node", stderr="Timeout (30s)", success=False)

    def _exec_compiled(
        self, code: str, lang: str, ext: str, compile_cmd: list, run_cmd: list | None = None
    ) -> ExecResult:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=ext, mode="w", delete=False)
        try:
            tmp.write(code)
            tmp.close()
            if run_cmd is not None:
                comp = subprocess.run(compile_cmd + [tmp.name], capture_output=True, text=True, timeout=30)
                if comp.returncode != 0:
                    return ExecResult(language=lang, stderr=comp.stderr, success=False)
                result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=30)
            else:
                result = subprocess.run(compile_cmd + [tmp.name], capture_output=True, text=True, timeout=30)
            return ExecResult(language=lang, stdout=result.stdout, stderr=result.stderr, success=result.returncode == 0)
        except FileNotFoundError:
            return ExecResult(language=lang, stderr=f"{lang} compiler not found", success=False)
        except subprocess.TimeoutExpired:
            return ExecResult(language=lang, stderr="Timeout (30s)", success=False)
        finally:
            try:
                os.unlink(tmp.name)
                out = tmp.name.replace(ext, "")
                if os.path.exists(out):
                    os.unlink(out)
            except OSError:
                pass

    def _exec_go(self, code: str) -> ExecResult:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".go", mode="w", delete=False, dir="/tmp")
        try:
            tmp.write(code)
            tmp.close()
            result = subprocess.run(["go", "run", tmp.name], capture_output=True, text=True, timeout=30)
            return ExecResult(language="go", stdout=result.stdout, stderr=result.stderr, success=result.returncode == 0)
        except FileNotFoundError:
            return ExecResult(language="go", stderr="Go not found (install: brew install go)", success=False)
        except subprocess.TimeoutExpired:
            return ExecResult(language="go", stderr="Timeout (30s)", success=False)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def _exec_rust(self, code: str) -> ExecResult:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".rs", mode="w", delete=False, dir="/tmp")
        out_bin = tmp.name.replace(".rs", "")
        try:
            tmp.write(code)
            tmp.close()
            comp = subprocess.run(["rustc", tmp.name, "-o", out_bin], capture_output=True, text=True, timeout=30)
            if comp.returncode != 0:
                return ExecResult(language="rust", stderr=comp.stderr, success=False)
            result = subprocess.run([out_bin], capture_output=True, text=True, timeout=30)
            return ExecResult(
                language="rust", stdout=result.stdout, stderr=result.stderr, success=result.returncode == 0
            )
        except FileNotFoundError:
            return ExecResult(
                language="rust",
                stderr="rustc not found (install: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh)",
                success=False,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(language="rust", stderr="Timeout (30s)", success=False)
        finally:
            for f in (tmp.name, out_bin):
                try:
                    os.unlink(f)
                except OSError:
                    pass

    def _exec_c(self, code: str) -> ExecResult:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False, dir="/tmp")
        out_bin = tmp.name.replace(".c", "")
        try:
            tmp.write(code)
            tmp.close()
            comp = subprocess.run(["cc", tmp.name, "-o", out_bin, "-lm"], capture_output=True, text=True, timeout=30)
            if comp.returncode != 0:
                return ExecResult(language="c", stderr=comp.stderr, success=False)
            result = subprocess.run([out_bin], capture_output=True, text=True, timeout=30)
            return ExecResult(language="c", stdout=result.stdout, stderr=result.stderr, success=result.returncode == 0)
        except FileNotFoundError:
            return ExecResult(
                language="c", stderr="cc not found (install Xcode CLT: xcode-select --install)", success=False
            )
        except subprocess.TimeoutExpired:
            return ExecResult(language="c", stderr="Timeout (30s)", success=False)
        finally:
            for f in (tmp.name, out_bin):
                try:
                    os.unlink(f)
                except OSError:
                    pass

    def _exec_cpp(self, code: str) -> ExecResult:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False, dir="/tmp")
        out_bin = tmp.name.replace(".cpp", "")
        try:
            tmp.write(code)
            tmp.close()
            comp = subprocess.run(
                ["c++", "-std=c++17", tmp.name, "-o", out_bin], capture_output=True, text=True, timeout=30
            )
            if comp.returncode != 0:
                return ExecResult(language="cpp", stderr=comp.stderr, success=False)
            result = subprocess.run([out_bin], capture_output=True, text=True, timeout=30)
            return ExecResult(
                language="cpp", stdout=result.stdout, stderr=result.stderr, success=result.returncode == 0
            )
        except FileNotFoundError:
            return ExecResult(
                language="cpp", stderr="c++ not found (install Xcode CLT: xcode-select --install)", success=False
            )
        except subprocess.TimeoutExpired:
            return ExecResult(language="cpp", stderr="Timeout (30s)", success=False)
        finally:
            for f in (tmp.name, out_bin):
                try:
                    os.unlink(f)
                except OSError:
                    pass

    def _exec_ocaml(self, code: str) -> ExecResult:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".ml", mode="w", delete=False, dir="/tmp")
        try:
            tmp.write(code)
            tmp.close()
            result = subprocess.run(["ocaml", tmp.name], capture_output=True, text=True, timeout=30)
            return ExecResult(
                language="ocaml", stdout=result.stdout, stderr=result.stderr, success=result.returncode == 0
            )
        except FileNotFoundError:
            return ExecResult(language="ocaml", stderr="OCaml not found (install: brew install ocaml)", success=False)
        except subprocess.TimeoutExpired:
            return ExecResult(language="ocaml", stderr="Timeout (30s)", success=False)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def _exec_lua(self, code: str) -> ExecResult:
        try:
            result = subprocess.run(["lua", "-e", code], capture_output=True, text=True, timeout=30)
            return ExecResult(
                language="lua", stdout=result.stdout, stderr=result.stderr, success=result.returncode == 0
            )
        except FileNotFoundError:
            return ExecResult(language="lua", stderr="Lua not found (install: brew install lua)", success=False)
        except subprocess.TimeoutExpired:
            return ExecResult(language="lua", stderr="Timeout (30s)", success=False)

    def _exec_typescript(self, code: str) -> ExecResult:
        for runner in (["npx", "tsx", "-e"], ["npx", "ts-node", "-e"], ["deno", "eval"]):
            try:
                result = subprocess.run(runner + [code], capture_output=True, text=True, timeout=30)
                return ExecResult(
                    language="typescript", stdout=result.stdout, stderr=result.stderr, success=result.returncode == 0
                )
            except FileNotFoundError:
                continue
            except subprocess.TimeoutExpired:
                return ExecResult(language="typescript", stderr="Timeout (30s)", success=False)
        return ExecResult(
            language="typescript", stderr="TypeScript runner not found (install: npm i -g tsx)", success=False
        )

    def _exec_latex(self, code: str) -> ExecResult:
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="morie_tex_")
        tex_path = os.path.join(tmpdir, "input.tex")
        try:
            with open(tex_path, "w") as f:
                f.write(code)
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, tex_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            pdf_path = os.path.join(tmpdir, "input.pdf")
            if os.path.exists(pdf_path):
                stdout = (
                    f"PDF generated: {pdf_path}\n{result.stdout[-500:]}"
                    if len(result.stdout) > 500
                    else f"PDF generated: {pdf_path}\n{result.stdout}"
                )
            else:
                stdout = result.stdout
            return ExecResult(language="latex", stdout=stdout, stderr=result.stderr, success=result.returncode == 0)
        except FileNotFoundError:
            return ExecResult(
                language="latex", stderr="pdflatex not found (install: brew install --cask mactex)", success=False
            )
        except subprocess.TimeoutExpired:
            return ExecResult(language="latex", stderr="Timeout (60s)", success=False)

    def _exec_psql(self, code: str) -> ExecResult:
        db_url = os.environ.get("MORIE_PGURL", os.environ.get("DATABASE_URL", ""))
        if db_url:
            cmd = ["psql", db_url, "-c", code]
        else:
            cmd = ["psql", "-c", code]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return ExecResult(
                language="psql", stdout=result.stdout, stderr=result.stderr, success=result.returncode == 0
            )
        except FileNotFoundError:
            return ExecResult(
                language="psql", stderr="psql not found (install: brew install postgresql)", success=False
            )
        except subprocess.TimeoutExpired:
            return ExecResult(language="psql", stderr="Timeout (30s)", success=False)

    def _exec_oneshot(self, code: str, lang: str, cmd: list, timeout: int = 30) -> ExecResult:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return ExecResult(language=lang, stdout=result.stdout, stderr=result.stderr, success=result.returncode == 0)
        except FileNotFoundError:
            return ExecResult(language=lang, stderr=f"{lang} not found", success=False)
        except subprocess.TimeoutExpired:
            return ExecResult(language=lang, stderr=f"Timeout ({timeout}s)", success=False)

    def _exec_file_based(self, code: str, lang: str, ext: str, run_cmd: list, timeout: int = 30) -> ExecResult:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=ext, mode="w", delete=False, dir="/tmp")
        try:
            tmp.write(code)
            tmp.close()
            result = subprocess.run(run_cmd + [tmp.name], capture_output=True, text=True, timeout=timeout)
            return ExecResult(language=lang, stdout=result.stdout, stderr=result.stderr, success=result.returncode == 0)
        except FileNotFoundError:
            return ExecResult(language=lang, stderr=f"{lang} not found", success=False)
        except subprocess.TimeoutExpired:
            return ExecResult(language=lang, stderr=f"Timeout ({timeout}s)", success=False)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def _exec_compile_run(self, code: str, lang: str, ext: str, compile_cmd: list, timeout: int = 30) -> ExecResult:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=ext, mode="w", delete=False, dir="/tmp")
        out_bin = tmp.name.rsplit(".", 1)[0]
        try:
            tmp.write(code)
            tmp.close()
            comp = subprocess.run(
                compile_cmd + [tmp.name, "-o", out_bin], capture_output=True, text=True, timeout=timeout
            )
            if comp.returncode != 0:
                return ExecResult(language=lang, stderr=comp.stderr, success=False)
            result = subprocess.run([out_bin], capture_output=True, text=True, timeout=timeout)
            return ExecResult(language=lang, stdout=result.stdout, stderr=result.stderr, success=result.returncode == 0)
        except FileNotFoundError:
            return ExecResult(language=lang, stderr=f"{lang} compiler not found", success=False)
        except subprocess.TimeoutExpired:
            return ExecResult(language=lang, stderr=f"Timeout ({timeout}s)", success=False)
        finally:
            for f in (tmp.name, out_bin):
                try:
                    os.unlink(f)
                except OSError:
                    pass

    def _exec_perl(self, code: str) -> ExecResult:
        return self._exec_oneshot(code, "perl", ["perl", "-e", code])

    def _exec_ruby(self, code: str) -> ExecResult:
        return self._exec_oneshot(code, "ruby", ["ruby", "-e", code])

    def _exec_php(self, code: str) -> ExecResult:
        if not code.strip().startswith("<?php"):
            code = "<?php " + code
        return self._exec_oneshot(code, "php", ["php", "-r", code])

    def _exec_java(self, code: str) -> ExecResult:
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="morie_java_")
        java_path = os.path.join(tmpdir, "Main.java")
        try:
            with open(java_path, "w") as f:
                f.write(code)
            comp = subprocess.run(["javac", java_path], capture_output=True, text=True, timeout=30)
            if comp.returncode != 0:
                return ExecResult(language="java", stderr=comp.stderr, success=False)
            result = subprocess.run(["java", "-cp", tmpdir, "Main"], capture_output=True, text=True, timeout=30)
            return ExecResult(
                language="java", stdout=result.stdout, stderr=result.stderr, success=result.returncode == 0
            )
        except FileNotFoundError:
            return ExecResult(language="java", stderr="Java not found (install JDK)", success=False)
        except subprocess.TimeoutExpired:
            return ExecResult(language="java", stderr="Timeout (30s)", success=False)

    def _exec_kotlin(self, code: str) -> ExecResult:
        return self._exec_file_based(code, "kotlin", ".kts", ["kotlinc", "-script"])

    def _exec_scala(self, code: str) -> ExecResult:
        return self._exec_file_based(code, "scala", ".scala", ["scala"])

    def _exec_swift(self, code: str) -> ExecResult:
        return self._exec_file_based(code, "swift", ".swift", ["swift"])

    def _exec_haskell(self, code: str) -> ExecResult:
        return self._exec_file_based(code, "haskell", ".hs", ["runhaskell"])

    def _exec_elixir(self, code: str) -> ExecResult:
        return self._exec_oneshot(code, "elixir", ["elixir", "-e", code])

    def _exec_zig(self, code: str) -> ExecResult:
        return self._exec_file_based(code, "zig", ".zig", ["zig", "run"])

    def _exec_nim(self, code: str) -> ExecResult:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".nim", mode="w", delete=False, dir="/tmp")
        try:
            tmp.write(code)
            tmp.close()
            result = subprocess.run(["nim", "r", "--hints:off", tmp.name], capture_output=True, text=True, timeout=30)
            return ExecResult(
                language="nim", stdout=result.stdout, stderr=result.stderr, success=result.returncode == 0
            )
        except FileNotFoundError:
            return ExecResult(language="nim", stderr="Nim not found (install: brew install nim)", success=False)
        except subprocess.TimeoutExpired:
            return ExecResult(language="nim", stderr="Timeout (30s)", success=False)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def _exec_d(self, code: str) -> ExecResult:
        return self._exec_file_based(code, "d", ".d", ["rdmd"])

    def _exec_fortran(self, code: str) -> ExecResult:
        return self._exec_compile_run(code, "fortran", ".f90", ["gfortran"])

    def _exec_octave(self, code: str) -> ExecResult:
        return self._exec_oneshot(code, "octave", ["octave", "--eval", code])

    def _exec_prolog(self, code: str) -> ExecResult:
        return self._exec_file_based(code, "prolog", ".pl", ["swipl", "-g", "main", "-t", "halt", "-s"])

    def _exec_scheme(self, code: str) -> ExecResult:
        for runner in (["racket", "-e"], ["guile", "-c"], ["chicken-csi", "-e"]):
            try:
                result = subprocess.run(runner + [code], capture_output=True, text=True, timeout=30)
                return ExecResult(
                    language="scheme", stdout=result.stdout, stderr=result.stderr, success=result.returncode == 0
                )
            except FileNotFoundError:
                continue
            except subprocess.TimeoutExpired:
                return ExecResult(language="scheme", stderr="Timeout (30s)", success=False)
        return ExecResult(language="scheme", stderr="Scheme not found (install: brew install racket)", success=False)

    def _exec_clojure(self, code: str) -> ExecResult:
        return self._exec_file_based(code, "clojure", ".clj", ["clojure"])

    def _exec_dart(self, code: str) -> ExecResult:
        return self._exec_file_based(code, "dart", ".dart", ["dart", "run"])

    def _exec_powershell(self, code: str) -> ExecResult:
        for ps in ("pwsh", "powershell"):
            try:
                result = subprocess.run([ps, "-Command", code], capture_output=True, text=True, timeout=30)
                return ExecResult(
                    language="powershell", stdout=result.stdout, stderr=result.stderr, success=result.returncode == 0
                )
            except FileNotFoundError:
                continue
            except subprocess.TimeoutExpired:
                return ExecResult(language="powershell", stderr="Timeout (30s)", success=False)
        return ExecResult(
            language="powershell", stderr="PowerShell not found (install: brew install powershell)", success=False
        )

    def _exec_awk(self, code: str) -> ExecResult:
        return self._exec_oneshot(code, "awk", ["awk", code])

    def _exec_rmd(self, code: str) -> ExecResult:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".Rmd", mode="w", delete=False, dir="/tmp")
        try:
            tmp.write(code)
            tmp.close()
            result = subprocess.run(
                ["Rscript", "-e", f'rmarkdown::render("{tmp.name}", quiet=TRUE)'],
                capture_output=True,
                text=True,
                timeout=120,
            )
            html_path = tmp.name.replace(".Rmd", ".html")
            if os.path.exists(html_path):
                stdout = f"Rendered: {html_path}"
            else:
                stdout = result.stdout
            return ExecResult(language="rmd", stdout=stdout, stderr=result.stderr, success=result.returncode == 0)
        except FileNotFoundError:
            return ExecResult(language="rmd", stderr="Rscript not found", success=False)
        except subprocess.TimeoutExpired:
            return ExecResult(language="rmd", stderr="Timeout (120s)", success=False)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def _bridge_variables(self, from_lang: str, variables: dict) -> None:
        for name, value in variables.items():
            if from_lang != "python" and value is not None:
                self._py_ns[name] = value

            if from_lang != "r" and self._r_proc and self._r_proc.poll() is None:
                self._inject_r_var(name, value)

    def _inject_r_var(self, name: str, value: object) -> None:
        if value is None:
            return
        if isinstance(value, bool):
            r_val = "TRUE" if value else "FALSE"
        elif isinstance(value, (int, float)):
            r_val = str(value)
        elif isinstance(value, str):
            r_val = f'"{value}"'
        elif isinstance(value, (list, tuple)):
            try:
                r_val = f"c({','.join(str(v) for v in value)})"
            except Exception:
                return
        else:
            return
        try:
            sentinel = f"__INJ_{id(name)}__"
            self._r_proc.stdin.write(f"{name} <- {r_val}\ncat('{sentinel}\\n')\n")
            self._r_proc.stdin.flush()
            while True:
                raw = self._r_proc.stdout.readline()
                if not raw or sentinel in raw:
                    break
        except Exception:
            pass

    def close(self) -> None:
        for proc in (self._r_proc, self._julia_proc, self._node_proc):
            if proc and proc.poll() is None:
                proc.terminate()
        if self._db_conn:
            self._db_conn.close()
            self._db_conn = None

    def available_languages(self) -> dict[str, bool]:
        result = {"python": True}
        for name, cmd in [
            ("r", "R"),
            ("julia", "julia"),
            ("node", "node"),
            ("shell", self._shell),
            ("go", "go"),
            ("rust", "rustc"),
            ("c", "cc"),
            ("cpp", "c++"),
            ("ocaml", "ocaml"),
            ("lua", "lua"),
            ("typescript", "npx"),
            ("latex", "pdflatex"),
            ("psql", "psql"),
            ("perl", "perl"),
            ("ruby", "ruby"),
            ("php", "php"),
            ("java", "javac"),
            ("kotlin", "kotlinc"),
            ("scala", "scala"),
            ("swift", "swift"),
            ("haskell", "runhaskell"),
            ("elixir", "elixir"),
            ("zig", "zig"),
            ("nim", "nim"),
            ("d", "rdmd"),
            ("fortran", "gfortran"),
            ("octave", "octave"),
            ("prolog", "swipl"),
            ("scheme", "racket"),
            ("clojure", "clojure"),
            ("dart", "dart"),
            ("powershell", "pwsh"),
            ("awk", "awk"),
        ]:
            try:
                subprocess.run([cmd, "--version"], capture_output=True, timeout=5)
                result[name] = True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                result[name] = False
        result["sql"] = True
        result["rmd"] = result.get("r", False)
        return result


def run_headless_repl(
    polyglot: bool = True,
    auto_detect: bool = True,
    lang: str = "python",
) -> int:
    engine = PolyglotEngine(polyglot=polyglot, auto_detect=auto_detect)
    if not auto_detect:
        engine._default_lang = lang

    avail = engine.available_languages()
    langs = [k for k, v in avail.items() if v]
    print(f"MORIE Polyglot REPL -- {len(langs)} languages: {', '.join(langs)}")
    if polyglot:
        print("Polyglot mode ON -- variables bridge automatically across languages")
    print("Prefixes: R> J> Q> N> Go> Rs> C> C+> ML> Lu> TS> TX> PG> ! (shell)")
    print(f"Auto-detect: {'ON' if auto_detect else 'OFF'} | Default: {lang}")

    try:
        from morie.fn._registry import REGISTRY

        print(f"morie.fn loaded ({len(REGISTRY)} functions)")
    except ImportError:
        pass
    print("ctrl+d to exit\n")

    try:
        import readline
        import rlcompleter

        readline.set_completer(rlcompleter.Completer(engine._py_ns).complete)
        readline.parse_and_bind("tab: complete")
    except ImportError:
        pass

    while True:
        try:
            prompt = f"{LABELS.get(engine._default_lang, '[P]')} "
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line.strip():
            continue

        if line.strip() == "/quit" or line.strip() == "/exit":
            break
        if line.strip() == "/langs":
            for k, v in engine.available_languages().items():
                status = "available" if v else "not found"
                print(f"  {LABELS[k]} {k}: {status}")
            continue
        if line.strip().startswith("/lang "):
            new_lang = line.strip().split(None, 1)[1].lower()
            if new_lang in LANGUAGES:
                engine._default_lang = new_lang
                print(f"Default language: {new_lang}")
            else:
                print(f"Unknown language: {new_lang}. Options: {', '.join(LANGUAGES)}")
            continue
        if line.strip() == "/polyglot":
            engine.polyglot = not engine.polyglot
            print(f"Polyglot: {'ON' if engine.polyglot else 'OFF'}")
            continue

        result = engine.execute(line)
        label = LABELS.get(result.language, "[?]")

        if result.stdout.strip():
            for out_line in result.stdout.strip().splitlines()[:100]:
                print(f"  {out_line}")
        if result.stderr.strip():
            for err_line in result.stderr.strip().splitlines()[:20]:
                print(f"  [err] {err_line}")
        if result.variables and engine.polyglot:
            bridged = [k for k in result.variables if result.variables[k] is not None]
            if bridged:
                print(f"  {label} bridged: {', '.join(bridged)}")

    engine.close()
    return 0
