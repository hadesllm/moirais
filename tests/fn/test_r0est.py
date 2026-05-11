"""Tests for morie.fn.r0est -- R0 via next-generation matrix."""

import pytest
from morie.fn.r0est import r0_next_generation


class TestR0Est:
    def test_sir(self):
        F = [[0.3]]
        V = [[0.1]]
        res = r0_next_generation(F, V)
        assert res.measure == "R0_NGM"
        assert res.estimate == pytest.approx(3.0)

    def test_2x2(self):
        F = [[0.3, 0.0], [0.0, 0.2]]
        V = [[0.1, 0.0], [0.0, 0.1]]
        res = r0_next_generation(F, V)
        assert res.estimate == pytest.approx(3.0)

    def test_singular_raises(self):
        with pytest.raises(ValueError):
            r0_next_generation([[1]], [[0]])
