"""Tests for the headless polyglot REPL engine."""
import os
import pytest
from morie.polyglot import PolyglotEngine, detect_language, ExecResult, LABELS, LANGUAGES


class TestDetectLanguage:
    def test_python_default(self):
        assert detect_language("x = 5") == "python"
        assert detect_language("print('hello')") == "python"
        assert detect_language("import numpy") == "python"
        assert detect_language("from morie import fn") == "python"

    def test_r_assignment(self):
        assert detect_language("x <- 5") == "r"

    def test_r_pipe(self):
        assert detect_language("df %>% filter(x > 0)") == "r"

    def test_r_library(self):
        assert detect_language("library(ggplot2)") == "r"

    def test_r_functions(self):
        assert detect_language("lm(y ~ x, data=df)") == "r"
        assert detect_language("t.test(x, y)") == "r"

    def test_shell_bang(self):
        assert detect_language("!ls -la") == "shell"

    def test_shell_dollar(self):
        assert detect_language("$echo hello") == "shell"

    def test_shell_commands(self):
        assert detect_language("git status") == "shell"
        assert detect_language("docker ps") == "shell"
        assert detect_language("pip install numpy") == "shell"
        assert detect_language("ls -la") == "shell"

    def test_julia_using(self):
        assert detect_language("using LinearAlgebra") == "julia"

    def test_julia_println(self):
        assert detect_language("println(42)") == "julia"

    def test_julia_types(self):
        assert detect_language("x::Int64 = 5") == "julia"

    def test_sql_select(self):
        assert detect_language("SELECT * FROM datasets") == "sql"

    def test_sql_pragma(self):
        assert detect_language("PRAGMA table_info(cpads)") == "sql"

    def test_node_console(self):
        assert detect_language("console.log('hello')") == "node"

    def test_node_const(self):
        assert detect_language("const x = 42") == "node"

    def test_prefix_r(self):
        assert detect_language("R> summary(x)") == "r"
        assert detect_language("r> head(df)") == "r"

    def test_prefix_julia(self):
        assert detect_language("J> rand(10)") == "julia"

    def test_prefix_sql(self):
        assert detect_language("Q> SELECT 1") == "sql"

    def test_prefix_node(self):
        assert detect_language("N> Math.PI") == "node"

    def test_go_detection(self):
        assert detect_language("package main") == "go"
        assert detect_language('fmt.Println("hello")') == "go"
        assert detect_language("func main() {") == "go"

    def test_rust_detection(self):
        assert detect_language('println!("hello")') == "rust"
        assert detect_language("let mut x = 5;") == "rust"
        assert detect_language("use std::io;") == "rust"

    def test_c_detection(self):
        assert detect_language('#include <stdio.h>') == "c"
        assert detect_language('printf("hello\\n");') == "c"

    def test_cpp_detection(self):
        assert detect_language("std::cout << 42;") == "cpp"
        assert detect_language("#include <iostream>") == "cpp"
        assert detect_language("vector<int> v;") == "cpp"

    def test_ocaml_detection(self):
        assert detect_language('Printf.printf "hello\\n"') == "ocaml"
        assert detect_language("List.map f xs") == "ocaml"
        assert detect_language("let () = print_endline ;;") == "ocaml"

    def test_lua_detection(self):
        assert detect_language("local x = 5") == "lua"
        assert detect_language("for k, v in pairs(t) do") == "lua"

    def test_typescript_detection(self):
        assert detect_language("interface Foo { x: number }") == "typescript"
        assert detect_language("const x: string = 'hi'") == "typescript"

    def test_latex_detection(self):
        assert detect_language("\\documentclass{article}") == "latex"
        assert detect_language("\\begin{equation}") == "latex"
        assert detect_language("\\frac{1}{2}") == "latex"

    def test_psql_detection(self):
        assert detect_language("\\dt") == "psql"
        assert detect_language("\\d+ users") == "psql"

    def test_prefix_go(self):
        assert detect_language("Go> fmt.Println(42)") == "go"

    def test_prefix_rust(self):
        assert detect_language("Rs> println!(42)") == "rust"

    def test_prefix_c(self):
        assert detect_language("C> printf(42)") == "c"

    def test_prefix_cpp(self):
        assert detect_language("C+> cout << 42") == "cpp"

    def test_prefix_ocaml(self):
        assert detect_language("ML> List.map f xs") == "ocaml"

    def test_prefix_lua(self):
        assert detect_language("Lu> print(42)") == "lua"

    def test_prefix_typescript(self):
        assert detect_language("TS> console.log(42)") == "typescript"

    def test_prefix_latex(self):
        assert detect_language("TX> \\frac{1}{2}") == "latex"

    def test_prefix_psql(self):
        assert detect_language("PG> SELECT 1") == "psql"

    def test_ambiguous_defaults_to_python(self):
        assert detect_language("x = [1, 2, 3]") == "python"
        assert detect_language("for i in range(10):") == "python"
        assert detect_language("def foo(): pass") == "python"

    def test_empty_defaults(self):
        assert detect_language("") == "python"
        assert detect_language("   ") == "python"


class TestLabels:
    def test_all_languages_have_labels(self):
        for lang in LANGUAGES:
            assert lang in LABELS


class TestPolyglotEngine:
    def test_python_exec(self):
        engine = PolyglotEngine(polyglot=False, auto_detect=False)
        result = engine.execute("2 + 2")
        assert result.language == "python"
        assert result.success
        engine.close()

    def test_python_assignment_bridged(self):
        engine = PolyglotEngine(polyglot=True, auto_detect=False)
        result = engine.execute("x = 42")
        assert result.success
        assert "x" in engine._py_ns
        assert engine._py_ns["x"] == 42
        engine.close()

    def test_python_import(self):
        engine = PolyglotEngine(polyglot=False)
        result = engine.execute("import math; print(math.pi)")
        assert result.success
        assert "3.14" in result.stdout
        engine.close()

    def test_python_multiline(self):
        engine = PolyglotEngine(polyglot=False)
        code = "for i in range(3):\n    print(i)"
        result = engine.execute(code)
        assert result.success
        assert "0" in result.stdout
        assert "2" in result.stdout
        engine.close()

    def test_python_error(self):
        engine = PolyglotEngine(polyglot=False)
        result = engine.execute("1/0")
        assert not result.success
        assert "ZeroDivision" in result.stderr
        engine.close()

    def test_shell_exec(self):
        engine = PolyglotEngine(polyglot=False, auto_detect=True)
        result = engine.execute("!echo hello_polyglot")
        assert result.language == "shell"
        assert result.success
        assert "hello_polyglot" in result.stdout
        engine.close()

    def test_shell_variable_extraction(self):
        engine = PolyglotEngine(polyglot=True, auto_detect=True)
        result = engine.execute("!MY_VAR=42")
        assert result.language == "shell"
        assert "MY_VAR" in result.variables
        engine.close()

    def test_sql_select(self):
        engine = PolyglotEngine(polyglot=False, auto_detect=True)
        result = engine.execute("SELECT 1 + 1 AS result")
        assert result.language == "sql"
        assert result.success
        assert "result" in result.stdout
        assert "2" in result.stdout
        engine.close()

    def test_sql_create_and_query(self):
        engine = PolyglotEngine(polyglot=False, auto_detect=True)
        engine.execute("CREATE TABLE test_tbl (id INTEGER, name TEXT)")
        engine.execute("INSERT INTO test_tbl VALUES (1, 'alice')")
        result = engine.execute("SELECT * FROM test_tbl")
        assert result.success
        assert "alice" in result.stdout
        engine.close()

    def test_auto_detect_toggle(self):
        engine = PolyglotEngine(polyglot=False, auto_detect=True)
        r1 = engine.execute("print('python code')")
        assert r1.language == "python"
        r2 = engine.execute("!echo shell_code")
        assert r2.language == "shell"
        r3 = engine.execute("SELECT 1")
        assert r3.language == "sql"
        engine.close()

    def test_strip_prefix(self):
        engine = PolyglotEngine()
        code, lang = engine.strip_prefix("R> summary(x)")
        assert lang == "r"
        assert code == "summary(x)"
        code, lang = engine.strip_prefix("Q> SELECT 1")
        assert lang == "sql"
        assert code == "SELECT 1"
        code, lang = engine.strip_prefix("print(1)")
        assert lang is None
        engine.close()

    def test_available_languages(self):
        engine = PolyglotEngine()
        avail = engine.available_languages()
        assert avail["python"] is True
        assert avail["sql"] is True
        assert "r" in avail
        assert "shell" in avail
        engine.close()

    def test_exec_result_dataclass(self):
        r = ExecResult(language="python", stdout="42", success=True)
        assert r.language == "python"
        assert r.stdout == "42"
        assert r.success
        assert r.variables == {}


class TestPolyglotBridging:
    def test_python_var_available_in_namespace(self):
        engine = PolyglotEngine(polyglot=True)
        engine.execute("bridge_test = 99")
        assert engine._py_ns.get("bridge_test") == 99
        engine.close()

    def test_shell_var_bridges_to_python(self):
        engine = PolyglotEngine(polyglot=True, auto_detect=True)
        result = engine.execute("!SHELL_X=123")
        assert result.variables.get("SHELL_X") == 123
        assert engine._py_ns.get("SHELL_X") == 123
        engine.close()


@pytest.mark.skipif(
    not any(
        os.path.exists(p)
        for p in ["/usr/bin/R", "/usr/local/bin/R", "/Library/Frameworks/R.framework/Resources/bin/R"]
    ),
    reason="R not installed",
)
class TestPolyglotR:
    def test_r_exec(self):
        engine = PolyglotEngine(polyglot=False, auto_detect=True)
        result = engine.execute('R> cat(2 + 2, "\\n")')
        assert result.language == "r"
        assert result.success
        assert "4" in result.stdout
        engine.close()

    def test_r_assignment_bridges_to_python(self):
        engine = PolyglotEngine(polyglot=True, auto_detect=True)
        result = engine.execute("R> my_r_val <- 77")
        assert result.language == "r"
        assert engine._py_ns.get("my_r_val") == 77
        engine.close()


@pytest.mark.skipif(
    not any(os.path.exists(p) for p in ["/usr/bin/node", "/usr/local/bin/node", "/opt/homebrew/bin/node"]),
    reason="Node.js not installed",
)
class TestPolyglotNode:
    def test_node_exec(self):
        engine = PolyglotEngine(polyglot=False, auto_detect=True)
        result = engine.execute("N> console.log(2 + 2)")
        assert result.language == "node"
        assert result.success
        assert "4" in result.stdout
        engine.close()


import shutil


@pytest.mark.skipif(not shutil.which("go"), reason="Go not installed")
class TestPolyglotGo:
    def test_go_exec(self):
        engine = PolyglotEngine(polyglot=False)
        code = 'Go> package main\nimport "fmt"\nfunc main() { fmt.Println(42) }'
        result = engine.execute(code)
        assert result.language == "go"
        assert result.success
        assert "42" in result.stdout
        engine.close()


@pytest.mark.skipif(not shutil.which("rustc"), reason="Rust not installed")
class TestPolyglotRust:
    def test_rust_exec(self):
        engine = PolyglotEngine(polyglot=False)
        code = 'Rs> fn main() { println!("42"); }'
        result = engine.execute(code)
        assert result.language == "rust"
        assert result.success
        assert "42" in result.stdout
        engine.close()


@pytest.mark.skipif(not shutil.which("cc"), reason="C compiler not installed")
class TestPolyglotC:
    def test_c_exec(self):
        engine = PolyglotEngine(polyglot=False)
        code = 'C> #include <stdio.h>\nint main() { printf("42\\n"); return 0; }'
        result = engine.execute(code)
        assert result.language == "c"
        assert result.success
        assert "42" in result.stdout
        engine.close()


@pytest.mark.skipif(not shutil.which("c++"), reason="C++ compiler not installed")
class TestPolyglotCpp:
    def test_cpp_exec(self):
        engine = PolyglotEngine(polyglot=False)
        code = 'C+> #include <iostream>\nint main() { std::cout << 42 << std::endl; return 0; }'
        result = engine.execute(code)
        assert result.language == "cpp"
        assert result.success
        assert "42" in result.stdout
        engine.close()


@pytest.mark.skipif(not shutil.which("ocaml"), reason="OCaml not installed")
class TestPolyglotOCaml:
    def test_ocaml_exec(self):
        engine = PolyglotEngine(polyglot=False)
        code = "ML> print_int 42;; print_newline ();;"
        result = engine.execute(code)
        assert result.language == "ocaml"
        assert result.success
        assert "42" in result.stdout
        engine.close()


@pytest.mark.skipif(not shutil.which("lua"), reason="Lua not installed")
class TestPolyglotLua:
    def test_lua_exec(self):
        engine = PolyglotEngine(polyglot=False)
        code = "Lu> print(42)"
        result = engine.execute(code)
        assert result.language == "lua"
        assert result.success
        assert "42" in result.stdout
        engine.close()
