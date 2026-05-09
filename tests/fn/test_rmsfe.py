"""Tests for moirais.fn.rmsfe -- RMSFE."""
import numpy as np
import pytest
from moirais.fn.rmsfe import rmsfe_calc


class TestRMSFE:
    def test_perfect(self):
        a = np.array([1.0, 2.0, 3.0])
        res = rmsfe_calc(a, a)
        assert res.value == 0.0

    def test_basic(self):
        a = np.array([1.0, 2.0, 3.0])
        f = np.array([1.1, 2.2, 2.8])
        res = rmsfe_calc(a, f)
        assert res.value > 0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            rmsfe_calc(np.ones(3), np.ones(5))

    def test_cheatsheet(self):
        from moirais.fn.rmsfe import cheatsheet
        assert isinstance(cheatsheet(), str)
