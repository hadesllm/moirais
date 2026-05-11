"""Tests for morie.fn.eqgni — Gini coefficient."""

import pytest
import numpy as np
from morie.fn.eqgni import gini_coefficient
from morie.fn._containers import ESRes


class TestGini:
    def test_perfect_equality(self):
        r = gini_coefficient([10, 10, 10, 10])
        assert isinstance(r, ESRes)
        assert r.estimate == pytest.approx(0.0, abs=0.01)

    def test_inequality(self):
        r = gini_coefficient([1, 1, 1, 1, 1, 1, 1, 1, 1, 100])
        assert r.estimate > 0.5

    def test_too_few(self):
        with pytest.raises(ValueError):
            gini_coefficient([1])
