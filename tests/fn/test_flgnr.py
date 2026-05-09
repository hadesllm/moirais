"""Tests for fligner_killeen."""
import numpy as np, pytest
from moirais.fn.flgnr import fligner_killeen

class TestFligner:
    def test_equal_var(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 30)
        b = rng.normal(0, 1, 30)
        r = fligner_killeen(a, b)
        assert r.p_value > 0.05

    def test_unequal_var(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0, 1, 50)
        b = rng.normal(0, 5, 50)
        r = fligner_killeen(a, b)
        assert r.p_value < 0.05
