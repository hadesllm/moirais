"""Tests for morie.fn.rope -- Region of Practical Equivalence."""

import numpy as np
import pytest
from morie.fn.rope import rope_test


class TestROPE:
    def test_samples_inside_rope_accept(self):
        """Samples entirely within ROPE -> accept."""
        samples = np.linspace(-0.01, 0.01, 1000)
        result = rope_test(samples, rope_low=-0.1, rope_high=0.1)
        assert result["decision"] == "accept"
        assert result["proportion_in_rope"] == pytest.approx(1.0)

    def test_samples_outside_rope_reject(self):
        """Samples entirely above ROPE -> reject."""
        samples = np.linspace(5.0, 6.0, 1000)
        result = rope_test(samples, rope_low=-0.1, rope_high=0.1)
        assert result["decision"] == "reject"
        assert result["proportion_in_rope"] == pytest.approx(0.0)

    def test_overlapping_undecided(self):
        """Wide posterior overlapping ROPE -> undecided."""
        rng = np.random.default_rng(42)
        samples = rng.normal(0.05, 0.2, 5000)
        result = rope_test(samples, rope_low=-0.1, rope_high=0.1)
        assert result["decision"] == "undecided"

    def test_invalid_rope_raises(self):
        with pytest.raises(ValueError, match="less than"):
            rope_test([1, 2, 3, 4], rope_low=0.5, rope_high=-0.5)
