# morie.fn -- function file (hadesllm/morie)
"""Gemma 4 native function calling via local Ollama."""

from __future__ import annotations

from ._containers import DescriptiveResult
from ._richresult import RichResult


def gemma_function_call(
    prompt: str,
    tools: list[dict] | None = None,
    model: str = "perseus:e2b",
    temperature: float = 0.1,
    base_url: str = "http://localhost:11434",
) -> DescriptiveResult:
    """Call Gemma 4's native function calling via Ollama.

    Sends a prompt with tool definitions to Gemma 4 and returns the model's
    response including any tool calls it wants to make. Gemma 4 supports
    native tool use -- it can decide which function to call and with what
    arguments based on the user's natural language request.

    :param prompt: Natural language request (e.g. "compute dnorm(0)").
    :param tools: List of tool definitions in Ollama format. If None, uses
        a default set of morie statistical functions.
    :param model: Ollama model name. Default ``perseus:e2b``.
    :param temperature: Sampling temperature. Default 0.1.
    :param base_url: Ollama API endpoint. Default ``http://localhost:11434``.
    :return: DescriptiveResult with response text and any tool_calls in extra.
    """

    try:
        import httpx
    except ImportError:
        return DescriptiveResult(
            name="httpx not installed",
            extra={"error": "pip install httpx"},
        )

    if tools is None:
        tools = _default_morie_tools()

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 4096},
    }

    try:
        resp = httpx.post(f"{base_url}/api/chat", json=payload, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return DescriptiveResult(
            name=f"Ollama request failed: {e}",
            extra={"error": str(e)},
        )

    msg = data.get("message", {})
    content = msg.get("content", "") or msg.get("thinking", "")
    tool_calls = msg.get("tool_calls", [])

    results = []
    for tc in tool_calls:
        fn_name = tc.get("function", {}).get("name", "")
        fn_args = tc.get("function", {}).get("arguments", {})
        result = _execute_morie_function(fn_name, fn_args)
        results.append({"function": fn_name, "arguments": fn_args, "result": result})

    return DescriptiveResult(
        name=content if content else "Function call(s) returned",
        extra={
            "response": content,
            "tool_calls": tool_calls,
            "executed_results": results,
            "model": model,
        },
    )


def _execute_morie_function(name: str, args: dict):
    """Execute an morie fn/ function by name with given arguments."""
    try:
        from morie.fn._registry import REGISTRY

        if name in REGISTRY:
            mod = __import__(f"morie.fn.{name}", fromlist=[name])
            fn = getattr(mod, REGISTRY[name].function_name)
            return fn(**args)
        return RichResult(payload={"error": f"Function '{name}' not in registry"})
    except Exception as e:
        return RichResult(payload={"error": str(e)})


def _default_morie_tools() -> list[dict]:
    """Default set of morie tools for Gemma function calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "dnorm",
                "description": "Normal distribution PDF. Returns density at x for N(mean, sd^2).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "Quantile"},
                        "mean": {"type": "number", "description": "Mean (default 0)"},
                        "sd": {"type": "number", "description": "Std dev (default 1)"},
                    },
                    "required": ["x"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ate",
                "description": "Average Treatment Effect via IPW or OLS.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "Path to CSV file"},
                        "outcome": {"type": "string", "description": "Outcome column name"},
                        "treatment": {"type": "string", "description": "Treatment column name"},
                    },
                    "required": ["data", "outcome", "treatment"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "t2smp",
                "description": "Two-sample t-test. Returns t-statistic, p-value, CI.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "array", "items": {"type": "number"}, "description": "Sample 1"},
                        "y": {"type": "array", "items": {"type": "number"}, "description": "Sample 2"},
                        "equal_var": {"type": "boolean", "description": "Assume equal variance (default True)"},
                    },
                    "required": ["x", "y"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "d",
                "description": "Cohen's d effect size between two groups.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "array", "items": {"type": "number"}, "description": "Group 1"},
                        "y": {"type": "array", "items": {"type": "number"}, "description": "Group 2"},
                    },
                    "required": ["x", "y"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "luke",
                "description": "Summarize a dataset (descriptive statistics for all columns).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "Path to CSV or dataset key"},
                    },
                    "required": ["data"],
                },
            },
        },
    ]


def cheatsheet() -> str:
    return "gemma_function_call({}) -> Gemma 4 native function calling via local Ollama."
