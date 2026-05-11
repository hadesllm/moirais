"""Tests for morie.fn.lzwcd — LZW compression."""

import numpy as np
import pytest

from morie.fn.lzwcd import lzwcd


class TestLzwcd:
    def test_roundtrip(self):
        data = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.int64)
        result = lzwcd(data, alphabet_size=4)
        np.testing.assert_array_equal(result["decompressed"], data)

    def test_constant_sequence_compresses(self):
        data = np.zeros(100, dtype=np.int64)
        result = lzwcd(data, alphabet_size=2)
        assert result["compression_ratio"] > 1.0

    def test_random_less_compression(self):
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, size=200)
        result = lzwcd(data)
        assert result["compression_ratio"] >= 0.5

    def test_empty_data(self):
        result = lzwcd(np.array([], dtype=np.int64))
        assert result["compressed"] == []

    def test_out_of_range_error(self):
        with pytest.raises(ValueError):
            lzwcd(np.array([0, 256]), alphabet_size=256)

    def test_dictionary_grows(self):
        data = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        result = lzwcd(data, alphabet_size=2)
        assert result["dictionary_size"] > 2
