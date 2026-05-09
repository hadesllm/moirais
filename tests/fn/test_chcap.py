"""Tests for moirais.fn.chcap — channel capacity."""

import numpy as np
import pytest

from moirais.fn.chcap import chcap


class TestChcap:
    def test_noiseless_channel(self):
        W = np.eye(2)
        result = chcap(W)
        assert result["capacity"] == pytest.approx(1.0, abs=0.01)

    def test_useless_channel(self):
        W = np.array([[0.5, 0.5], [0.5, 0.5]])
        result = chcap(W)
        assert result["capacity"] == pytest.approx(0.0, abs=0.01)

    def test_bsc_half(self):
        W = np.array([[0.5, 0.5], [0.5, 0.5]])
        result = chcap(W)
        assert result["capacity"] == pytest.approx(0.0, abs=0.01)

    def test_optimal_input_sums_to_one(self):
        W = np.array([[0.9, 0.1], [0.2, 0.8]])
        result = chcap(W)
        assert np.isclose(result["optimal_input"].sum(), 1.0)

    def test_invalid_rows(self):
        with pytest.raises(ValueError):
            chcap(np.array([[0.5, 0.3], [0.5, 0.5]]))

    def test_capacity_nonnegative(self):
        W = np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2]])
        result = chcap(W)
        assert result["capacity"] >= 0
