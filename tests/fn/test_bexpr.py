"""Tests for bexpr (boolean expression evaluator)."""
from morie.fn.bexpr import boolean_eval


def test_boolean_eval_and():
    r = boolean_eval("A & B", {"A": 1, "B": 1})
    assert r.extra["result_bool"] is True


def test_boolean_eval_or():
    r = boolean_eval("A | B", {"A": 0, "B": 1})
    assert r.extra["result_bool"] is True
