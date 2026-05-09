"""Tests for moirais.fn.eqthl — Theil index."""

import pytest
from moirais.fn.eqthl import theil_index
from moirais.fn._containers import ESRes


class TestTheil:
    def test_equality(self):
        r = theil_index([10, 10, 10, 10])
        assert r.estimate == pytest.approx(0.0, abs=0.01)

    def test_inequality(self):
        r = theil_index([1, 1, 1, 100])
        assert r.estimate > 0
