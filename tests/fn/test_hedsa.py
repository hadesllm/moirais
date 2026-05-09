"""Tests for moirais.fn.hedsa -- DSA."""

import pytest
from moirais.fn.hedsa import deterministic_sensitivity


class TestDSA:
    def test_basic(self):
        res = deterministic_sensitivity(
            base_result=50000,
            param_name="cost",
            param_range=[800, 1000, 1200],
            results_at_range=[40000, 50000, 60000],
        )
        assert res.value == pytest.approx(20000.0)
        assert res.extra["param_name"] == "cost"


def test_cheatsheet():
    from moirais.fn.hedsa import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
