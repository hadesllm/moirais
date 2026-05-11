"""Tests for morie.fn.kolmc — Kolmogorov complexity approximation."""

import numpy as np
import pytest

from morie.fn.kolmc import kolmc


class TestKolmc:
    def test_constant_low_complexity(self):
        data = np.zeros(1000, dtype=np.int32)
        result = kolmc(data)
        assert result["compression_ratio"] > 5.0

    def test_random_high_complexity(self):
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, size=1000, dtype=np.uint8)
        result = kolmc(data)
        assert result["normalized_complexity"] > 0.5

    def test_empty(self):
        result = kolmc(np.array([]))
        assert result["compressed_size"] == 0
        assert result["compression_ratio"] == 1.0

    def test_normalized_between_0_and_2(self):
        data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result = kolmc(data)
        assert result["normalized_complexity"] > 0

    def test_complexity_bits_positive(self):
        data = np.array([10, 20, 30], dtype=np.int64)
        result = kolmc(data)
        assert result["complexity_bits"] > 0

    def test_output_keys(self):
        result = kolmc(np.array([1, 2, 3]))
        assert "compressed_size" in result
        assert "original_size" in result
        assert "compression_ratio" in result
