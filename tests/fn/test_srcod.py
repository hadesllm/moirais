"""Tests for morie.fn.srcod -- AST depth analysis."""

from morie.fn.srcod import ast_depth, srcod
from morie.fn._containers import DescriptiveResult


class TestSrcod:
    def test_alias(self):
        assert srcod is ast_depth

    def test_simple(self):
        result = ast_depth("x = 1")
        assert isinstance(result, DescriptiveResult)
        assert result.value >= 2

    def test_function(self):
        code = "def foo(x):\n    if x > 0:\n        return x\n    return -x"
        result = ast_depth(code)
        assert result.extra["n_functions"] == 1
        assert result.extra["n_branches"] >= 1
