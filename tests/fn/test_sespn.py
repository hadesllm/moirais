"""Tests for morie.fn.sespn -- Simple exponential smoothing."""
import numpy as np
import pytest
from morie.fn.sespn import ses


class TestSES:
    def test_basic(self):
        y = np.array([10, 12, 11, 13, 14, 15, 13, 12])
        res = ses(y, alpha=0.3)
        assert res.name == "ses"
        assert len(res.extra["fitted"]) == 8

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            ses(np.ones(10), alpha=1.5)

    def test_cheatsheet(self):
        from morie.fn.sespn import cheatsheet
        assert isinstance(cheatsheet(), str)
