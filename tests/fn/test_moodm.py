"""Tests for mood_median_test."""
import numpy as np, pytest
from morie.fn.moodm import mood_median_test

class TestMood:
    def test_different(self):
        a = np.array([1,2,3,4,5], dtype=float)
        b = np.array([10,11,12,13,14], dtype=float)
        r = mood_median_test(a, b)
        assert r.test_name == "Mood median"
        assert r.p_value < 0.05

    def test_same(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 30)
        b = rng.normal(0, 1, 30)
        r = mood_median_test(a, b)
        assert r.p_value > 0.01
