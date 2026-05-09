"""Tests for mcnemar_test."""
import numpy as np, pytest
from moirais.fn.mcnmr import mcnemar_test

class TestMcNemar:
    def test_basic(self):
        before = np.array([1,1,1,0,0,0,0,0,1,1])
        after  = np.array([1,1,0,0,1,1,1,0,1,0])
        r = mcnemar_test(before, after)
        assert r.test_name == "McNemar"
        assert 0 <= r.p_value <= 1

    def test_no_change(self):
        x = np.array([0,1,0,1,0,1])
        r = mcnemar_test(x, x)
        assert r.p_value == 1.0
