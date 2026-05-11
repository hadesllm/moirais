"""Tests for tetrachoric_corr."""
import pytest
from morie.fn.tetrc import tetrachoric_corr

class TestTetrachoric:
    def test_positive(self):
        r = tetrachoric_corr(50, 10, 10, 50)
        assert r.estimate > 0.5

    def test_negative(self):
        r = tetrachoric_corr(10, 50, 50, 10)
        assert r.estimate < -0.3
