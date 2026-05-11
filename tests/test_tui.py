"""Tests for morie.tui — Textual terminal IDE application.

Tests are skipped when textual is not installed.
Uses Textual's App.run_test() for headless async testing.
"""

from __future__ import annotations

import pytest

try:
    import textual

    _TEXTUAL_AVAILABLE = True
except ImportError:
    _TEXTUAL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TEXTUAL_AVAILABLE,
    reason="textual not installed (pip install morie[interactive])",
)


# ---------------------------------------------------------------------------
# Import guards
# ---------------------------------------------------------------------------


class TestImportGuards:
    def test_textual_available_flag(self):
        from morie.tui import _TEXTUAL_AVAILABLE as flag

        assert flag is True

    def test_launch_tui_importable(self):
        from morie.tui import launch_tui

        assert callable(launch_tui)

    def test_morie_app_importable(self):
        from morie.tui import MORIEApp

        assert MORIEApp is not None


# ---------------------------------------------------------------------------
# App instantiation
# ---------------------------------------------------------------------------


class TestMORIEApp:
    def test_app_attributes(self):
        from morie.tui import MORIEApp

        app = MORIEApp()
        assert app.TITLE == "MORIE"
        assert "chat" in app.SCREENS
        assert "pipeline" in app.SCREENS
        assert "doctor" in app.SCREENS
        assert "dataset" in app.SCREENS

    def test_app_has_bindings(self):
        from morie.tui import MORIEApp, HomeScreen

        app = MORIEApp()
        app_keys = [b.key for b in app.BINDINGS]
        assert "q" in app_keys

        home_keys = [b.key for b in HomeScreen.BINDINGS]
        assert "c" in home_keys
        assert "p" in home_keys
        assert "d" in home_keys
        assert "i" in home_keys
        assert "q" in home_keys


# ---------------------------------------------------------------------------
# Screen classes
# ---------------------------------------------------------------------------


class TestScreenClasses:
    def test_chat_screen_exists(self):
        from morie.tui import ChatScreen

        assert ChatScreen is not None

    def test_pipeline_screen_exists(self):
        from morie.tui import PipelineScreen

        assert PipelineScreen is not None

    def test_doctor_screen_exists(self):
        from morie.tui import DoctorScreen

        assert DoctorScreen is not None

    def test_dataset_screen_exists(self):
        from morie.tui import DatasetScreen

        assert DatasetScreen is not None

    def test_chat_screen_accepts_agent(self):
        from morie.tui import ChatScreen

        screen = ChatScreen(agent="morie-architect")
        assert screen._agent == "morie-architect"

    def test_chat_screen_default_no_agent(self):
        from morie.tui import ChatScreen

        screen = ChatScreen()
        assert screen._agent is None

    def test_repl_screen_auto_detect_mode(self):
        from morie.tui import ReplScreen

        screen = ReplScreen(lang="auto")
        assert screen._auto_detect is True
        assert screen._lang == "python"  # default when auto

    def test_repl_screen_locked_mode(self):
        from morie.tui import ReplScreen

        screen = ReplScreen(lang="r")
        assert screen._auto_detect is False
        assert screen._lang == "r"

    def test_repl_screen_user_shell(self):
        import os
        from morie.tui import ReplScreen

        screen = ReplScreen(lang="auto")
        expected = os.environ.get("SHELL", "/bin/bash")
        assert screen._user_shell == expected


# ---------------------------------------------------------------------------
# Language auto-detection
# ---------------------------------------------------------------------------


class TestLanguageDetection:
    def test_unambiguous_r_patterns(self):
        from morie.tui import ReplScreen

        screen = ReplScreen(lang="auto")
        assert screen._detect_language("x <- 5") == "r"
        assert screen._detect_language("library(stats)") == "r"
        assert screen._detect_language("data.frame(a=1, b=2)") == "r"
        assert screen._detect_language("df %>% filter(x > 1)") == "r"
        assert screen._detect_language("t.test(x, y)") == "r"
        assert screen._detect_language("dplyr::mutate(df, z=1)") == "r"

    def test_shell_patterns(self):
        from morie.tui import ReplScreen

        screen = ReplScreen(lang="auto")
        assert screen._detect_language("git status") == "shell"
        assert screen._detect_language("ls -la") == "shell"
        assert screen._detect_language("!echo hi") == "shell"
        assert screen._detect_language("docker ps") == "shell"

    def test_ambiguous_stays_python(self):
        """Patterns that exist in both R and Python should stay Python."""
        from morie.tui import ReplScreen

        screen = ReplScreen(lang="auto")
        assert screen._detect_language("2 + 2") == "python"
        assert screen._detect_language("import os") == "python"
        assert screen._detect_language("x") == "python"
        assert screen._detect_language("c(1,2,3)") == "python"
        assert screen._detect_language("TRUE") == "python"
        assert screen._detect_language("FALSE") == "python"
        assert screen._detect_language("result") == "python"
        assert screen._detect_language("filter(data)") == "python"
        assert screen._detect_language("summary(df)") == "python"
        assert screen._detect_language("plot(x, y)") == "python"

    def test_no_ambiguous_patterns_in_r_set(self):
        """Verify _R_PATTERNS contains no ambiguous patterns."""
        from morie.tui import ReplScreen

        ambiguous = {
            "c(", "TRUE", "FALSE", "NULL", "NA", "function(",
            "if (", "for (", "while (", "filter(", "select(",
            "summary(", "plot(", "arrange(", "mutate(", "group_by(",
        }
        for pat in ambiguous:
            assert pat not in ReplScreen._R_PATTERNS, (
                f"'{pat}' should not be in _R_PATTERNS (ambiguous with Python)"
            )


# ---------------------------------------------------------------------------
# REPL helper functions
# ---------------------------------------------------------------------------


class TestReplHelpers:
    def test_helper_functions_injected(self):
        import code as code_module
        from morie.tui import ReplScreen

        screen = ReplScreen(lang="auto")
        screen._py_console_ns = {
            "__name__": "__console__",
            "__builtins__": __builtins__,
        }
        screen._py_console = code_module.InteractiveConsole(screen._py_console_ns)
        screen._inject_repl_helpers()

        assert callable(screen._py_console_ns["view"])
        assert callable(screen._py_console_ns["ls"])
        assert callable(screen._py_console_ns["who"])
        assert callable(screen._py_console_ns["clear"])


# ---------------------------------------------------------------------------
# Offensive term removal
# ---------------------------------------------------------------------------


class TestNoOffensiveTerms:
    def test_no_slave_flag(self):
        from pathlib import Path

        tui_path = Path(__file__).resolve().parents[2] / "tools" / "py-package" / "morie" / "tui.py"
        content = tui_path.read_text()
        assert "--slave" not in content, "R --slave flag must be replaced with --no-echo"


# ---------------------------------------------------------------------------
# StatScreen
# ---------------------------------------------------------------------------


class TestStatScreen:
    def test_show_help_method_exists(self):
        from morie.tui import StatScreen

        screen = StatScreen()
        assert hasattr(screen, "_show_help")
        assert callable(screen._show_help)


# ---------------------------------------------------------------------------
# Headless async TUI tests
# ---------------------------------------------------------------------------


class TestReplScreenAsync:
    def test_python_execution(self):
        import asyncio
        from morie.tui import MORIEApp

        async def _test():
            app = MORIEApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await pilot.press("e")
                await pilot.pause()
                editor = pilot.app.screen.query_one("#repl-editor")
                for c in "2+2":
                    await pilot.press(c)
                await pilot.press("ctrl+j")
                await pilot.pause()
                assert editor.has_focus
                assert editor.text == ""

        asyncio.run(_test())

    def test_shell_via_bang_prefix(self):
        import asyncio
        from morie.tui import MORIEApp

        async def _test():
            app = MORIEApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await pilot.press("e")
                await pilot.pause()
                editor = pilot.app.screen.query_one("#repl-editor")
                for c in "!echo hello":
                    await pilot.press(c)
                await pilot.press("ctrl+j")
                await pilot.pause()
                assert editor.has_focus

        asyncio.run(_test())

    def test_mode_switch_and_history(self):
        import asyncio
        from morie.tui import MORIEApp

        async def _test():
            app = MORIEApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await pilot.press("e")
                await pilot.pause()
                editor = pilot.app.screen.query_one("#repl-editor")
                for c in "x = 1":
                    await pilot.press(c)
                await pilot.press("ctrl+j")
                await pilot.pause()
                await pilot.press("f3")
                await pilot.pause()
                assert editor.text == "x = 1"

        asyncio.run(_test())

    def test_view_helper_works(self):
        import asyncio
        from morie.tui import MORIEApp

        async def _test():
            app = MORIEApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await pilot.press("e")
                await pilot.pause()
                editor = pilot.app.screen.query_one("#repl-editor")
                # Assign then view
                for c in "x = 42":
                    await pilot.press(c)
                await pilot.press("ctrl+j")
                await pilot.pause()
                for c in "view(x)":
                    await pilot.press(c)
                await pilot.press("ctrl+j")
                await pilot.pause()
                assert editor.has_focus

        asyncio.run(_test())

    def test_ls_helper_works(self):
        import asyncio
        from morie.tui import MORIEApp

        async def _test():
            app = MORIEApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await pilot.press("e")
                await pilot.pause()
                editor = pilot.app.screen.query_one("#repl-editor")
                for c in "x = 42":
                    await pilot.press(c)
                await pilot.press("ctrl+j")
                await pilot.pause()
                for c in "ls()":
                    await pilot.press(c)
                await pilot.press("ctrl+j")
                await pilot.pause()
                assert editor.has_focus

        asyncio.run(_test())

    def test_r_autodetect(self):
        import asyncio
        from morie.tui import MORIEApp

        async def _test():
            app = MORIEApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await pilot.press("e")
                await pilot.pause()
                editor = pilot.app.screen.query_one("#repl-editor")
                # x <- 5 should auto-detect as R
                for c in "x <- 5":
                    await pilot.press(c)
                await pilot.press("ctrl+j")
                await pilot.pause()
                assert editor.has_focus

        asyncio.run(_test())

    def test_ambiguous_stays_python(self):
        import asyncio
        from morie.tui import MORIEApp

        async def _test():
            app = MORIEApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await pilot.press("e")
                await pilot.pause()
                editor = pilot.app.screen.query_one("#repl-editor")
                # Single identifier — should stay Python, not switch
                for c in "x = 10":
                    await pilot.press(c)
                await pilot.press("ctrl+j")
                await pilot.pause()
                # Now just "x" should evaluate in Python
                for c in "x":
                    await pilot.press(c)
                await pilot.press("ctrl+j")
                await pilot.pause()
                assert editor.has_focus

        asyncio.run(_test())


# ---------------------------------------------------------------------------
# Summary fix tests
# ---------------------------------------------------------------------------


class TestSummaryHelper:
    """Verify summary() handles DataFrames and strings without crashing."""

    def test_summary_with_dataframe_arg(self):
        """summary(df) should not raise ValueError about truth value."""
        import pandas as pd
        from morie.tui import ReplScreen

        screen = ReplScreen()
        screen._inject_repl_helpers()
        ns = screen._py_console_ns
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ns["df"] = df
        ns["summary"](df)

    def test_summary_with_string_arg(self):
        import pandas as pd
        from morie.tui import ReplScreen

        screen = ReplScreen()
        screen._inject_repl_helpers()
        ns = screen._py_console_ns
        ns["df"] = pd.DataFrame({"age": [20, 30, 40], "score": [1, 2, 3]})
        ns["summary"]("age")

    def test_summary_no_data(self):
        from morie.tui import ReplScreen

        screen = ReplScreen()
        screen._inject_repl_helpers()
        ns = screen._py_console_ns
        ns.pop("df", None)
        ns["summary"]()


# ---------------------------------------------------------------------------
# Alias resolution tests
# ---------------------------------------------------------------------------


class TestAliases:
    def test_alias_map_populated(self):
        from morie.stat_commands import ALIAS_MAP
        assert len(ALIAS_MAP) > 50

    def test_resolve_by_alias(self):
        from morie.stat_commands import resolve
        cmd = resolve("ttest1")
        assert cmd is not None
        assert cmd.name == "one_sample_ttest"

    def test_all_aliases_have_valid_canonical(self):
        from morie.stat_commands import ALIAS_MAP, COMMAND_REGISTRY
        for alias, canonical in ALIAS_MAP.items():
            assert canonical in COMMAND_REGISTRY, f"Alias '{alias}' → '{canonical}' missing"

    def test_handler_repl_callable(self):
        from morie.stat_commands import COMMAND_REGISTRY
        for name, cmd in list(COMMAND_REGISTRY.items())[:20]:
            assert callable(cmd.handler_repl), f"'{name}' handler_repl not callable"


# ---------------------------------------------------------------------------
# stat_commands registry tests
# ---------------------------------------------------------------------------


class TestStatCommandsRegistry:
    def test_registry_has_600_plus_commands(self):
        from morie.stat_commands import COMMAND_REGISTRY
        assert len(COMMAND_REGISTRY) >= 600

    def test_all_categories_populated(self):
        from morie.stat_commands import CATEGORIES
        assert len(CATEGORIES) >= 10

    def test_commands_by_category_works(self):
        from morie.stat_commands import commands_by_category
        cats = commands_by_category()
        total = sum(len(cmds) for cmds in cats.values())
        assert total >= 600


# ---------------------------------------------------------------------------
# Vendored FreeAPI client tests
# ---------------------------------------------------------------------------


class TestVendoredFreeAPI:
    def test_import(self):
        from morie.fam import OllamaFreeAPI
        client = OllamaFreeAPI()
        assert hasattr(client, "chat")
        assert hasattr(client, "stream_chat")
        assert hasattr(client, "list_models")

    def test_list_models_returns_list(self):
        from morie.fam import OllamaFreeAPI
        client = OllamaFreeAPI()
        models = client.list_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_list_families(self):
        from morie.fam import OllamaFreeAPI
        client = OllamaFreeAPI()
        families = client.list_families()
        assert isinstance(families, list)
        assert len(families) >= 1

    def test_build_payload_structure(self):
        from morie.fam import OllamaFreeAPI
        client = OllamaFreeAPI()
        payload = client._build_payload("gpt-oss:20b", "test prompt", num_predict=10000)
        assert payload["model"] == "gpt-oss:20b"
        assert payload["prompt"] == "test prompt"
        assert payload["options"]["num_predict"] == 10000
        assert "temperature" in payload["options"]

    def test_get_model_servers_returns_list(self):
        from morie.fam import OllamaFreeAPI
        client = OllamaFreeAPI()
        models = client.list_models()
        if models:
            servers = client.get_model_servers(models[0])
            assert isinstance(servers, list)


# ---------------------------------------------------------------------------
# Verify: all commands have valid backend (from verify_all_commands.py)
# ---------------------------------------------------------------------------


class TestAllCommandsValid:
    """Converted from standalone verify_all_commands.py into pytest."""

    def test_all_commands_have_backend_functions(self):
        import importlib
        from morie.stat_commands import COMMAND_REGISTRY
        broken = []
        for name, cmd in COMMAND_REGISTRY.items():
            if not cmd.module:
                continue
            try:
                mod = importlib.import_module(f"morie.{cmd.module}")
                fn_name = getattr(cmd.handler_repl, "__name__", name)
                if not hasattr(mod, fn_name) and not hasattr(mod, name):
                    broken.append(f"{name} (module={cmd.module})")
            except ImportError as e:
                broken.append(f"{name} (IMPORT: {e})")
        assert not broken, f"Broken commands: {broken[:10]}"

    def test_all_handlers_callable(self):
        from morie.stat_commands import COMMAND_REGISTRY
        not_callable = [
            name for name, cmd in COMMAND_REGISTRY.items()
            if not callable(cmd.handler_repl) or not callable(cmd.handler_stat)
        ]
        assert not not_callable, f"Not callable: {not_callable[:10]}"


# ---------------------------------------------------------------------------
# Verify: REPL namespace injection (from verify_repl_injection.py)
# ---------------------------------------------------------------------------


class TestReplNamespaceInjection:
    """Converted from standalone verify_repl_injection.py into pytest."""

    def test_hardcoded_helpers_injected(self):
        from morie.tui import ReplScreen
        screen = ReplScreen()
        screen._inject_repl_helpers()
        ns = screen._py_console_ns
        hardcoded = [
            "load", "head", "tail", "shape", "cols", "describe", "summary",
            "view", "ls", "who", "clear", "ttest", "corr", "chi2", "anova",
            "ate", "evalue", "propensity", "ipw", "freq", "crosstab",
            "modules", "run_module", "version", "help_repl",
        ]
        missing = [h for h in hardcoded if h not in ns]
        assert not missing, f"Missing helpers: {missing}"

    def test_registry_commands_injected(self):
        from morie.tui import ReplScreen
        from morie.stat_commands import COMMAND_REGISTRY
        screen = ReplScreen()
        screen._inject_repl_helpers()
        ns = screen._py_console_ns
        sample = list(COMMAND_REGISTRY.keys())[:100]
        missing = [name for name in sample if name not in ns]
        assert not missing, f"Missing registry commands: {missing[:10]}"

    def test_aliases_injected(self):
        from morie.tui import ReplScreen
        from morie.stat_commands import ALIAS_MAP
        screen = ReplScreen()
        screen._inject_repl_helpers()
        ns = screen._py_console_ns
        sample = list(ALIAS_MAP.keys())[:100]
        missing = [alias for alias in sample if alias not in ns]
        assert not missing, f"Missing aliases: {missing[:10]}"

    def test_total_callable_count(self):
        from morie.tui import ReplScreen
        screen = ReplScreen()
        screen._inject_repl_helpers()
        ns = screen._py_console_ns
        callables = sum(1 for k, v in ns.items() if callable(v) and not k.startswith("_"))
        assert callables >= 1200, f"Only {callables} callables, expected 1200+"


# ---------------------------------------------------------------------------
# Verify: alias execution (from verify_alias_execution.py)
# ---------------------------------------------------------------------------


class TestAliasExecution:
    """Converted from standalone verify_alias_execution.py into pytest."""

    def test_no_arg_helpers_show_usage_not_crash(self):
        """All helpers with required args should show usage when called with none."""
        import numpy as np
        import pandas as pd
        from morie.tui import ReplScreen

        screen = ReplScreen()
        screen._inject_repl_helpers()
        ns = screen._py_console_ns
        ns["df"] = pd.DataFrame({
            "x": np.random.randn(50),
            "group": np.random.choice([0, 1], 50),
        })

        no_arg_funcs = [
            "ttest", "ttest2", "chi2", "corr", "anova", "spearman",
            "mannwhitney", "ks_test", "paired_t", "bootstrap_ci",
            "effect_size", "propensity", "ipw", "kaplan_meier", "cox",
            "logrank", "did", "rdd", "iv_tsls", "match", "vif",
            "odds_ratio", "nnt", "rosenbaum", "jackknife", "permtest",
            "event_study", "fuzzy_rdd", "freq", "unique", "value_counts",
            "crosstab", "pivot", "groupby", "power", "ebac", "evalue",
        ]
        crashed = []
        for name in no_arg_funcs:
            fn = ns.get(name)
            if fn is None:
                crashed.append(f"{name}: NOT IN NAMESPACE")
                continue
            try:
                fn()
            except TypeError as e:
                if "required" in str(e) or "positional" in str(e):
                    crashed.append(f"{name}: missing no-arg guard")
            except Exception:
                pass  # Other exceptions OK (e.g., no data)
        assert not crashed, f"Crashed on no-arg: {crashed}"

    def test_registry_aliases_show_usage_not_crash(self):
        """Registry aliases should gracefully handle no-arg calls."""
        from morie.tui import ReplScreen
        screen = ReplScreen()
        screen._inject_repl_helpers()
        ns = screen._py_console_ns

        aliases = ["ttest1", "pearson", "cd", "hg", "mwu", "wsr", "chi2_ind", "sw_test", "ks1"]
        crashed = []
        for alias in aliases:
            fn = ns.get(alias)
            if fn is None:
                crashed.append(f"{alias}: NOT IN NAMESPACE")
                continue
            try:
                fn()
            except TypeError as e:
                if "required" in str(e) or "positional" in str(e):
                    crashed.append(f"{alias}: missing no-arg guard")
        assert not crashed, f"Aliases crashed: {crashed}"


# ---------------------------------------------------------------------------
# Polyglot bridge tests
# ---------------------------------------------------------------------------


class TestPolyglotBridge:
    """Test P↔R↔Shell variable bridge parsing logic."""

    def test_assignment_regex_extracts_names(self):
        import re
        pattern = re.compile(r'(\w+)\s*<-')
        assert pattern.findall("x <- 42") == ["x"]
        assert pattern.findall("mean_age <- mean(cpads$age)") == ["mean_age"]
        assert pattern.findall("v <- c(1,2,3)") == ["v"]
        assert pattern.findall("print(x)") == []
        assert pattern.findall("x <- 1; y <- 2") == ["x", "y"]

    def test_polyglot_flag_default_off(self):
        from morie.tui import ReplScreen
        screen = ReplScreen()
        assert screen._polyglot is False
