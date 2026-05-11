"""Tests for morie.fn.eqplm — Palma ratio."""

import pytest
import numpy as np
from morie.fn.eqplm import palma_ratio
from morie.fn._containers import ESRes


class TestPalma:
    def test_equal(self):
        r = palma_ratio([10] * 100)
        assert r.estimate == pytest.approx(0.25, abs=0.05)

    def test_unequal(self):
        vals = [1] * 90 + [100] * 10
        r = palma_ratio(vals)
        assert r.estimate > 5
