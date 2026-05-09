"""Tests for moirais.fn.zands -- Zivot-Andrews test."""
import numpy as np
import pytest
from moirais.fn.zands import za_test


class TestZA:
    def test_basic(self):
        rng = np.random.default_rng(42)
        y = np.concatenate([rng.standard_normal(50), rng.standard_normal(50) + 3])
        res = za_test(y)
        assert "break_index" in res.extra

    def test_short_raises(self):
        with pytest.raises(ValueError):
            za_test(np.ones(10))

    def test_cheatsheet(self):
        from moirais.fn.zands import cheatsheet
        assert isinstance(cheatsheet(), str)
