"""Tests for morie.fn.dexpn -- Double exponential smoothing."""
import numpy as np
import pytest
from morie.fn.dexpn import des


class TestDES:
    def test_basic(self):
        y = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28.0])
        res = des(y, alpha=0.3, beta=0.1)
        assert res.name == "des"
        assert len(res.extra["level"]) == 10

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            des(np.ones(10), alpha=0, beta=0.1)

    def test_cheatsheet(self):
        from morie.fn.dexpn import cheatsheet
        assert isinstance(cheatsheet(), str)
