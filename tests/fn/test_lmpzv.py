"""Tests for morie.fn.lmpzv — Lempel-Ziv complexity."""

import numpy as np
import pytest

from morie.fn.lmpzv import lmpzv


class TestLmpzv:
    def test_constant_sequence(self):
        seq = np.zeros(100, dtype=int)
        result = lmpzv(seq)
        assert result["complexity"] <= 3

    def test_alternating(self):
        seq = np.array([0, 1] * 50)
        result = lmpzv(seq)
        assert result["complexity"] > 1

    def test_random_higher(self):
        rng = np.random.default_rng(42)
        seq = rng.integers(0, 2, size=200)
        result = lmpzv(seq)
        assert result["complexity"] > 10

    def test_empty(self):
        result = lmpzv(np.array([]))
        assert result["complexity"] == 0
        assert result["length"] == 0

    def test_single_element(self):
        result = lmpzv(np.array([1]))
        assert result["complexity"] >= 1

    def test_normalized_near_one_for_random(self):
        rng = np.random.default_rng(42)
        seq = rng.integers(0, 2, size=1000)
        result = lmpzv(seq)
        assert 0.5 < result["normalized"] < 2.0
