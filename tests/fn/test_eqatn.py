"""Tests for morie.fn.eqatn — Atkinson index."""

import pytest
import numpy as np
from morie.fn.eqatn import atkinson_index
from morie.fn._containers import ESRes


class TestAtkinson:
    def test_equality(self):
        r = atkinson_index([10, 10, 10, 10])
        assert r.estimate == pytest.approx(0.0, abs=0.01)

    def test_inequality(self):
        r = atkinson_index([1, 1, 1, 100])
        assert r.estimate > 0

    def test_epsilon_1(self):
        r = atkinson_index([1, 2, 3, 4], epsilon=1.0)
        assert isinstance(r, ESRes)
        assert 0 <= r.estimate <= 1
