"""Tests for moirais.fn.cohrc -- Coherence function."""
import numpy as np
import pytest
from moirais.fn.cohrc import coherence


class TestCoherence:
    def test_identical(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(64)
        res = coherence(x, x)
        assert res.extra["coherence"] is not None

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            coherence(np.ones(10), np.ones(15))

    def test_cheatsheet(self):
        from moirais.fn.cohrc import cheatsheet
        assert isinstance(cheatsheet(), str)
