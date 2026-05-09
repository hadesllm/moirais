"""Tests for fizzbuzz_stats."""
import pytest
from moirais.fn.fizz import fizzbuzz_stats

class TestFizz:
    def test_15(self):
        r = fizzbuzz_stats(15)
        assert r.extra["fizzbuzz_count"] == 1
        assert r.extra["fizz_count"] == 4
        assert r.extra["buzz_count"] == 2

    def test_proportions(self):
        r = fizzbuzz_stats(300)
        assert r.extra["fizzbuzz_pct"] == pytest.approx(100/15, abs=1)
