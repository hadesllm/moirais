"""Tests for moirais.agent — Perseus agentic loop with tool calling."""

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict
from unittest.mock import patch

from moirais.agent import (
    AgentResponse,
    PerseusAgent,
    tool_describe_data,
    tool_execute_code,
    tool_get_cheatsheet,
    tool_inspect_error,
    tool_list_files,
    tool_read_file,
    tool_run_moirais_function,
    tool_run_shell,
    tool_search_codebase,
    tool_search_functions,
    tool_write_file,
)


class TestToolReadFile:

    def test_read_existing_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("line one\nline two\n")
        result = tool_read_file(str(f), sandbox=tmp_path)
        assert "line one" in result
        assert "line two" in result

    def test_read_missing_file_returns_error(self, tmp_path):
        result = tool_read_file(str(tmp_path / "nope.txt"), sandbox=tmp_path)
        assert "not found" in result.lower()

    def test_sandbox_rejects_outside_path(self, tmp_path):
        result = tool_read_file("/etc/passwd", sandbox=tmp_path)
        assert "outside" in result.lower()

    def test_read_binary_file_graceful(self, tmp_path):
        f = tmp_path / "bin.dat"
        f.write_bytes(b"\x00\x01\x02\xff")
        result = tool_read_file(str(f), sandbox=tmp_path)
        assert isinstance(result, str)

    def test_sandbox_rejects_traversal(self, tmp_path):
        evil = str(tmp_path / ".." / ".." / "etc" / "passwd")
        result = tool_read_file(evil, sandbox=tmp_path)
        assert "outside" in result.lower()


class TestToolWriteFile:

    def test_write_creates_file(self, tmp_path):
        target = tmp_path / "out.txt"
        result = tool_write_file(str(target), "hello world", sandbox=tmp_path)
        assert target.exists()
        assert target.read_text() == "hello world"
        assert "error" not in result.lower()

    def test_write_creates_subdirectory(self, tmp_path):
        target = tmp_path / "sub" / "deep" / "out.csv"
        tool_write_file(str(target), "a,b\n1,2\n", sandbox=tmp_path)
        assert target.exists()
        assert target.read_text() == "a,b\n1,2\n"

    def test_write_overwrites_existing(self, tmp_path):
        target = tmp_path / "exist.txt"
        target.write_text("old")
        tool_write_file(str(target), "new", sandbox=tmp_path)
        assert target.read_text() == "new"

    def test_sandbox_rejects_outside_path(self, tmp_path):
        result = tool_write_file("/tmp/agent_evil_test.txt", "pwned", sandbox=tmp_path)
        assert "outside" in result.lower()

    def test_write_returns_confirmation(self, tmp_path):
        target = tmp_path / "ok.txt"
        result = tool_write_file(str(target), "data", sandbox=tmp_path)
        assert "wrote" in result.lower()


class TestToolExecuteCode:

    def test_simple_expression(self):
        result = tool_execute_code("print(2 + 3)")
        assert "5" in result

    def test_multiline_code(self):
        code = textwrap.dedent("""\
            x = [1, 2, 3]
            total = sum(x)
            print(f"sum={total}")
        """)
        result = tool_execute_code(code)
        assert "sum=6" in result

    def test_syntax_error_captured(self):
        result = tool_execute_code("def f(:")
        assert "error" in result.lower() or "syntax" in result.lower()

    def test_runtime_error_captured(self):
        result = tool_execute_code("1/0")
        assert "zero" in result.lower() or "error" in result.lower()

    def test_import_numpy(self):
        result = tool_execute_code("import numpy as np; print(np.mean([1,2,3]))")
        assert "2.0" in result


class TestToolSearchFunctions:

    def test_search_ttest(self):
        result = tool_search_functions("t-test")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_search_effect_size(self):
        result = tool_search_functions("effect size")
        assert isinstance(result, str)

    def test_search_causal(self):
        result = tool_search_functions("causal")
        assert isinstance(result, str)

    def test_search_empty_query(self):
        result = tool_search_functions("")
        assert isinstance(result, str)

    def test_search_returns_function_names(self):
        result = tool_search_functions("normal distribution")
        assert "dnorm" in result.lower() or "normal" in result.lower()


class TestToolRunMoiraisFunction:

    def test_run_dnorm(self):
        result = tool_run_moirais_function("dnorm", {"x": 0.0, "mean": 0.0, "sd": 1.0})
        assert isinstance(result, str)
        assert "0.39" in result

    def test_run_unknown_function(self):
        result = tool_run_moirais_function("nonexistent_function_xyz")
        assert "not in registry" in result.lower()

    def test_run_with_no_args(self):
        result = tool_run_moirais_function("dnorm", {"x": 1.0})
        assert isinstance(result, str)

    def test_run_returns_string(self):
        result = tool_run_moirais_function("dnorm", {"x": 1.0})
        assert isinstance(result, str)


class TestToolListFiles:

    def test_list_directory(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.csv").write_text("")
        result = tool_list_files(str(tmp_path), sandbox=tmp_path)
        assert "a.py" in result
        assert "b.csv" in result

    def test_list_empty_directory(self, tmp_path):
        sub = tmp_path / "empty"
        sub.mkdir()
        result = tool_list_files(str(sub), sandbox=tmp_path)
        assert isinstance(result, str)

    def test_sandbox_rejects_outside_path(self, tmp_path):
        result = tool_list_files("/etc", sandbox=tmp_path)
        assert "outside" in result.lower()

    def test_list_nonexistent_returns_error(self, tmp_path):
        result = tool_list_files(str(tmp_path / "nope"), sandbox=tmp_path)
        assert "not a directory" in result.lower() or "error" in result.lower()


class TestToolInspectError:

    def test_inspect_returns_string(self):
        result = tool_inspect_error()
        assert isinstance(result, str)

    def test_inspect_no_crash(self):
        result = tool_inspect_error()
        assert "error" not in result.lower() or "traceback" in result.lower() or "no recent" in result.lower()


class TestAgentResponse:

    def test_fields_present(self):
        resp = AgentResponse(
            text="Here is the answer.",
            tool_calls_made=[{"name": "search_functions"}],
            iterations=2,
            model="perseus:e2b",
        )
        assert resp.text == "Here is the answer."
        assert resp.iterations == 2
        assert resp.model == "perseus:e2b"
        assert len(resp.tool_calls_made) == 1

    def test_dataclass_asdict(self):
        resp = AgentResponse(text="ok", tool_calls_made=[], iterations=1, model="local")
        d = asdict(resp)
        assert d["text"] == "ok"
        assert d["iterations"] == 1

    def test_empty_tool_calls(self):
        resp = AgentResponse(text="no tools", tool_calls_made=[], iterations=0, model="local")
        assert resp.tool_calls_made == []

    def test_serializable_to_json(self):
        resp = AgentResponse(
            text="result",
            tool_calls_made=[{"name": "read_file"}],
            iterations=1,
            model="gemma4:e2b",
        )
        serialized = json.dumps(asdict(resp))
        assert "result" in serialized


class TestPerseusAgent:

    def test_init_default(self):
        agent = PerseusAgent()
        assert agent._max_iterations > 0
        agent.close()

    def test_init_custom_max_iterations(self):
        agent = PerseusAgent(max_iterations=3)
        assert agent._max_iterations == 3
        agent.close()

    def test_init_with_sandbox_root(self, tmp_path):
        agent = PerseusAgent(sandbox_root=str(tmp_path))
        assert str(agent._sandbox) == str(tmp_path)
        agent.close()

    def test_system_prompt_includes_moirais(self):
        agent = PerseusAgent()
        prompt = agent._build_system_prompt()
        assert "moirais" in prompt.lower() or "Perseus" in prompt
        agent.close()

    def test_tool_definitions_built(self):
        agent = PerseusAgent()
        defs = agent._build_tool_definitions()
        assert isinstance(defs, list)
        assert len(defs) > 0
        names = {d["function"]["name"] for d in defs}
        assert "read_file" in names
        assert "execute_code" in names
        assert "search_functions" in names
        agent.close()

    @patch("moirais.agent.PerseusAgent._send_to_ollama")
    def test_chat_no_tool_calls(self, mock_llm):
        mock_llm.return_value = {
            "message": {
                "role": "assistant",
                "content": "The ATE is the average causal effect.",
                "tool_calls": [],
            },
        }
        agent = PerseusAgent()
        resp = agent.chat("What is ATE?")
        assert isinstance(resp, AgentResponse)
        assert "ATE" in resp.text or "average" in resp.text.lower()
        assert resp.tool_calls_made == []
        assert resp.iterations == 1
        agent.close()

    @patch("moirais.agent.PerseusAgent._send_to_ollama")
    def test_chat_with_tool_call(self, mock_llm):
        mock_llm.side_effect = [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_functions",
                                "arguments": {"query": "propensity"},
                            },
                        }
                    ],
                },
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "Found ipw and aipw functions for propensity score analysis.",
                    "tool_calls": [],
                },
            },
        ]
        agent = PerseusAgent()
        resp = agent.chat("Find propensity score functions")
        assert isinstance(resp, AgentResponse)
        assert any(tc["name"] == "search_functions" for tc in resp.tool_calls_made)
        assert resp.iterations >= 1
        agent.close()

    @patch("moirais.agent.PerseusAgent._send_to_ollama")
    def test_chat_respects_max_iterations(self, mock_llm):
        mock_llm.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_functions",
                            "arguments": {"query": "loop forever"},
                        },
                    }
                ],
            },
        }
        agent = PerseusAgent(max_iterations=2)
        resp = agent.chat("loop")
        assert resp.iterations <= 2
        agent.close()

    @patch("moirais.agent.PerseusAgent._send_to_ollama")
    def test_chat_returns_model_info(self, mock_llm):
        mock_llm.return_value = {
            "message": {
                "role": "assistant",
                "content": "hello",
                "tool_calls": [],
            },
        }
        agent = PerseusAgent(model="perseus:e2b")
        resp = agent.chat("hi")
        assert resp.model == "perseus:e2b"
        agent.close()


class TestToolGetCheatsheet:

    def test_known_function(self):
        result = tool_get_cheatsheet("dnorm")
        assert "dnorm" in result.lower() or "normal" in result.lower()

    def test_unknown_function(self):
        result = tool_get_cheatsheet("nonexistent_xyz")
        assert "error" in result.lower() or "no cheatsheet" in result.lower()


class TestToolSearchCodebase:

    def test_finds_pattern(self, tmp_path):
        src = tmp_path / "example.py"
        src.write_text("def moran_i(data):\n    pass\n")
        result = tool_search_codebase("moran_i", sandbox=tmp_path)
        assert "moran_i" in result

    def test_no_matches(self, tmp_path):
        src = tmp_path / "empty.py"
        src.write_text("pass\n")
        result = tool_search_codebase("zzz_nonexistent_pattern_xyz", sandbox=tmp_path)
        assert "No matches" in result


class TestToolRunShell:

    def test_basic_command(self):
        result = tool_run_shell("echo hello")
        assert "hello" in result

    def test_blocked_command(self):
        result = tool_run_shell("rm -rf /")
        assert "Blocked" in result

    def test_timeout(self):
        result = tool_run_shell("sleep 60", timeout=1)
        assert "timed out" in result.lower()


class TestToolDescribeData:

    def test_basic_dataframe(self):
        result = tool_describe_data("df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})")
        assert "Shape" in result
        assert "(3, 2)" in result

    def test_bad_code(self):
        result = tool_describe_data("raise ValueError('nope')")
        assert "Error" in result or "ValueError" in result


class TestToolDispatchNewTools:

    def test_dispatch_has_all_tools(self):
        from moirais.agent import _CORE_TOOLS, _TOOL_DISPATCH
        tool_names = {t["function"]["name"] for t in _CORE_TOOLS}
        dispatch_names = set(_TOOL_DISPATCH.keys())
        assert tool_names == dispatch_names, f"Mismatch: defined={tool_names - dispatch_names}, dispatch={dispatch_names - tool_names}"

    def test_tool_count_is_20(self):
        from moirais.agent import _CORE_TOOLS
        assert len(_CORE_TOOLS) == 20


class TestFreeAPIAgent:

    def test_parse_tool_calls_valid(self):
        from moirais.agent import FreeAPIAgent
        agent = FreeAPIAgent()
        text = 'Let me search. <tool_call>{"name": "search_functions", "arguments": {"query": "moran"}}</tool_call>'
        calls = agent._parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "search_functions"
        assert calls[0].arguments == {"query": "moran"}

    def test_parse_tool_calls_no_match(self):
        from moirais.agent import FreeAPIAgent
        agent = FreeAPIAgent()
        calls = agent._parse_tool_calls("No tool calls here.")
        assert calls == []

    def test_parse_tool_calls_invalid_json(self):
        from moirais.agent import FreeAPIAgent
        agent = FreeAPIAgent()
        calls = agent._parse_tool_calls("<tool_call>not json</tool_call>")
        assert calls == []

    def test_parse_tool_calls_unknown_tool_ignored(self):
        from moirais.agent import FreeAPIAgent
        agent = FreeAPIAgent()
        calls = agent._parse_tool_calls('<tool_call>{"name": "evil_tool", "arguments": {}}</tool_call>')
        assert calls == []

    def test_parse_multiple_tool_calls(self):
        from moirais.agent import FreeAPIAgent
        agent = FreeAPIAgent()
        text = (
            '<tool_call>{"name": "search_functions", "arguments": {"query": "t-test"}}</tool_call> '
            '<tool_call>{"name": "get_cheatsheet", "arguments": {"name": "dnorm"}}</tool_call>'
        )
        calls = agent._parse_tool_calls(text)
        assert len(calls) == 2

    @patch("moirais.agent.FreeAPIAgent._send")
    def test_chat_no_tool_calls(self, mock_send):
        from moirais.agent import FreeAPIAgent
        mock_send.return_value = "The ATE is the average treatment effect."
        agent = FreeAPIAgent()
        resp = agent.chat("What is ATE?")
        assert isinstance(resp, AgentResponse)
        assert "ATE" in resp.text or "average" in resp.text.lower()
        assert resp.tool_calls_made == []

    @patch("moirais.agent.FreeAPIAgent._send")
    def test_chat_with_tool_call(self, mock_send):
        from moirais.agent import FreeAPIAgent
        mock_send.side_effect = [
            'Let me search. <tool_call>{"name": "search_functions", "arguments": {"query": "propensity"}}</tool_call>',
            "Found ipw and aipw functions for propensity score analysis.",
        ]
        agent = FreeAPIAgent()
        resp = agent.chat("Find propensity score functions")
        assert isinstance(resp, AgentResponse)
        assert any(tc["name"] == "search_functions" for tc in resp.tool_calls_made)
        assert resp.iterations >= 1

    def test_create_agent_freeapi_provider(self):
        from moirais.agent import FreeAPIAgent, create_agent
        agent = create_agent(provider="freeapi")
        assert isinstance(agent, FreeAPIAgent)

    def test_text_tool_prompt_exists(self):
        from moirais.agent import _TEXT_TOOL_PROMPT
        assert "tool_call" in _TEXT_TOOL_PROMPT
        assert "search_functions" in _TEXT_TOOL_PROMPT


class TestDomainKnowledge:

    def test_domain_knowledge_has_20_domains(self):
        from moirais.agent import _DOMAIN_KNOWLEDGE
        assert len(_DOMAIN_KNOWLEDGE) >= 20

    def test_each_domain_has_core_functions(self):
        from moirais.agent import _DOMAIN_KNOWLEDGE
        for domain, info in _DOMAIN_KNOWLEDGE.items():
            assert "core" in info, f"{domain} missing core"
            assert len(info["core"]) >= 1, f"{domain} has empty core"

    def test_each_domain_has_workflows(self):
        from moirais.agent import _DOMAIN_KNOWLEDGE
        for domain, info in _DOMAIN_KNOWLEDGE.items():
            assert "workflows" in info, f"{domain} missing workflows"

    def test_domain_guide_spatial(self):
        from moirais.agent import tool_domain_guide
        result = tool_domain_guide("spatial")
        assert "SPATIAL" in result
        assert "moran" in result
        assert "Workflows" in result

    def test_domain_guide_causal(self):
        from moirais.agent import tool_domain_guide
        result = tool_domain_guide("causal")
        assert "CAUSAL" in result
        assert "ipw" in result

    def test_domain_guide_unknown_falls_back(self):
        from moirais.agent import tool_domain_guide
        result = tool_domain_guide("nonexistent_domain_xyz")
        assert "Unknown domain" in result or "Available" in result

    def test_domain_guide_substring_match(self):
        from moirais.agent import tool_domain_guide
        result = tool_domain_guide("signal")
        assert "biomedical" in result.lower() or "BIOMEDICAL" in result

    def test_recommend_analysis_spatial(self):
        from moirais.agent import tool_recommend_analysis
        result = tool_recommend_analysis("How do I compute spatial autocorrelation?")
        assert "spatial" in result.lower()

    def test_recommend_analysis_causal(self):
        from moirais.agent import tool_recommend_analysis
        result = tool_recommend_analysis("What is the average treatment effect?")
        assert "causal" in result.lower()

    def test_recommend_analysis_unknown(self):
        from moirais.agent import tool_recommend_analysis
        result = tool_recommend_analysis("xyzzy plugh")
        assert "search_functions" in result

    def test_category_tree(self):
        from moirais.agent import tool_category_tree
        result = tool_category_tree()
        assert "TAXONOMY" in result
        assert "categories" in result.lower()

    def test_similar_functions_existing(self):
        from moirais.agent import tool_similar_functions
        result = tool_similar_functions("ate")
        assert "Similar" in result or "similar" in result

    def test_similar_functions_missing(self):
        from moirais.agent import tool_similar_functions
        result = tool_similar_functions("nonexistent_xyz")
        assert "not found" in result.lower()

    def test_textbook_reference_spatial(self):
        from moirais.agent import tool_textbook_reference
        result = tool_textbook_reference("kriging")
        assert "Schabenberger" in result

    def test_textbook_reference_emg(self):
        from moirais.agent import tool_textbook_reference
        result = tool_textbook_reference("emg")
        assert "Rangayyan" in result

    def test_textbook_reference_unknown(self):
        from moirais.agent import tool_textbook_reference
        result = tool_textbook_reference("xyzzy")
        assert "No specific reference" in result

    def test_run_pipeline_valid(self):
        from moirais.agent import tool_run_pipeline
        import json
        steps = json.dumps([{"fn": "dnorm", "kwargs": {"x": 0.0}}])
        result = tool_run_pipeline(steps)
        assert "Step 1" in result
        assert "dnorm" in result

    def test_run_pipeline_invalid_json(self):
        from moirais.agent import tool_run_pipeline
        result = tool_run_pipeline("not json")
        assert "Invalid JSON" in result

    def test_run_pipeline_empty(self):
        from moirais.agent import tool_run_pipeline
        result = tool_run_pipeline("[]")
        assert "empty" in result.lower()

    def test_compare_methods_empty(self):
        from moirais.agent import tool_compare_methods
        result = tool_compare_methods("")
        assert "Provide" in result

    def test_run_suite_unknown_domain(self):
        from moirais.agent import tool_run_suite
        result = tool_run_suite("nonexistent_xyz")
        assert "Unknown domain" in result

    def test_run_suite_distributions(self):
        from moirais.agent import tool_run_suite
        result = tool_run_suite("distributions")
        assert "DISTRIBUTIONS" in result

    def test_dispatch_has_all_new_tools(self):
        from moirais.agent import _TOOL_DISPATCH
        new_tools = ["domain_guide", "recommend_analysis", "category_tree",
                     "similar_functions", "textbook_reference", "run_pipeline",
                     "compare_methods", "run_suite"]
        for tool in new_tools:
            assert tool in _TOOL_DISPATCH, f"{tool} missing from dispatch"

    def test_small_model_gets_8_tools(self):
        from moirais.agent import PerseusAgent
        agent = PerseusAgent(model="functiongemma:270m")
        tools = agent._build_tool_definitions()
        assert len(tools) == 8
        names = {t["function"]["name"] for t in tools}
        assert "domain_guide" in names
        assert "recommend_analysis" in names
        assert "run_pipeline" in names

    def test_large_model_gets_20_tools(self):
        from moirais.agent import PerseusAgent
        agent = PerseusAgent(model="perseus:e2b")
        tools = agent._build_tool_definitions()
        assert len(tools) == 20

    def test_system_prompt_mentions_demigod(self):
        from moirais.agent import _SYSTEM_PROMPT_FULL
        assert "demigod" in _SYSTEM_PROMPT_FULL

    def test_system_prompt_mentions_domains(self):
        from moirais.agent import _SYSTEM_PROMPT_FULL
        assert "domain_guide" in _SYSTEM_PROMPT_FULL
        assert "recommend_analysis" in _SYSTEM_PROMPT_FULL
        assert "run_suite" in _SYSTEM_PROMPT_FULL
