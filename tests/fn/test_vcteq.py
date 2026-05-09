"""Tests for moirais.fn.vcteq — victim equity."""

import pytest
from moirais.fn.vcteq import victim_equity
from moirais.fn._containers import DescriptiveResult


class TestVictimEquity:
    def test_basic(self):
        r = victim_equity({"A": 0.05, "B": 0.10}, {"A": 10000, "B": 10000})
        assert isinstance(r, DescriptiveResult)
        assert r.extra["most_victimized"] == "B"

    def test_too_few(self):
        with pytest.raises(ValueError):
            victim_equity({"A": 0.1}, {"A": 100})
