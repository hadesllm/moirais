"""Tests for morie.fn.barcd -- CSP backtracking solver."""

from morie.fn.barcd import csp_backtrack, barcd
from morie.fn._containers import DescriptiveResult


class TestBarcd:
    def test_alias(self):
        assert barcd is csp_backtrack

    def test_simple_csp(self):
        variables = ["a", "b"]
        domains = {"a": [1, 2], "b": [1, 2]}
        constraints = [("a", "b", lambda x, y: x != y)]
        r = csp_backtrack(variables, domains, constraints)
        assert isinstance(r, DescriptiveResult)
        assert r.value is not None
        assert r.value["a"] != r.value["b"]

    def test_no_solution(self):
        variables = ["a", "b"]
        domains = {"a": [1], "b": [1]}
        constraints = [("a", "b", lambda x, y: x != y)]
        r = csp_backtrack(variables, domains, constraints)
        assert r.value is None
