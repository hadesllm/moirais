"""Tests for morie.fn.typst — typical set."""

import numpy as np
import pytest

from morie.fn.typst import typst


class TestTypst:
    def test_binary_entropy(self):
        result = typst(np.array([0.5, 0.5]), 10)
        assert result["entropy"] == pytest.approx(1.0, abs=1e-10)

    def test_typical_prob_lower(self):
        result = typst(np.array([0.3, 0.7]), 100, epsilon=0.05)
        assert result["typical_prob_lower"] == pytest.approx(0.95)

    def test_size_bounds(self):
        result = typst(np.array([0.5, 0.5]), 10, epsilon=0.1)
        assert result["typical_set_size_lower"] <= result["typical_set_size_upper"]

    def test_log2_total(self):
        result = typst(np.array([0.25, 0.25, 0.25, 0.25]), 5)
        assert result["log2_total_sequences"] == pytest.approx(10.0)

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            typst(np.array([0.5, 0.5]), 0)

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError):
            typst(np.array([0.5, 0.5]), 10, epsilon=-0.1)
