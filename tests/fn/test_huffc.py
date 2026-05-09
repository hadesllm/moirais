"""Tests for moirais.fn.huffc — Huffman coding."""

import numpy as np
import pytest

from moirais.fn.huffc import huffc


class TestHuffc:
    def test_uniform_binary(self):
        result = huffc(np.array([0.5, 0.5]))
        assert result["avg_length"] == pytest.approx(1.0)
        assert result["entropy"] == pytest.approx(1.0, abs=1e-6)

    def test_prefix_free(self):
        result = huffc(np.array([0.5, 0.25, 0.125, 0.125]))
        codes = list(result["codebook"].values())
        for i, c1 in enumerate(codes):
            for j, c2 in enumerate(codes):
                if i != j:
                    assert not c2.startswith(c1)

    def test_efficiency_leq_1(self):
        result = huffc(np.array([0.4, 0.3, 0.2, 0.1]))
        assert result["efficiency"] <= 1.0 + 1e-10
        assert result["efficiency"] > 0.0

    def test_avg_length_geq_entropy(self):
        result = huffc(np.array([0.4, 0.3, 0.2, 0.1]))
        assert result["avg_length"] >= result["entropy"] - 1e-10

    def test_custom_symbols(self):
        result = huffc(np.array([0.7, 0.3]), symbols=["A", "B"])
        assert "A" in result["codebook"]
        assert "B" in result["codebook"]

    def test_single_symbol(self):
        result = huffc(np.array([1.0]))
        assert result["codebook"][0] == "0"

    def test_invalid_pmf(self):
        with pytest.raises(ValueError):
            huffc(np.array([0.3, 0.3]))
