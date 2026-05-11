"""Tests for morie.fn.reedm — Reed-Muller codes."""

import numpy as np
import pytest

from morie.fn.reedm import reedm


class TestReedm:
    def test_rm_1_3(self):
        result = reedm(1, 3)
        assert result["n"] == 8
        assert result["k"] == 4
        assert result["d_min"] == 4

    def test_rm_0_m_repetition(self):
        result = reedm(0, 4)
        assert result["k"] == 1
        assert result["d_min"] == 16
        assert np.all(result["generator"] == 1)

    def test_rm_m_m_full(self):
        result = reedm(3, 3)
        assert result["k"] == 8
        assert result["n"] == 8
        assert result["d_min"] == 1

    def test_generator_binary(self):
        G = reedm(1, 4)["generator"]
        assert np.all((G == 0) | (G == 1))

    def test_rate(self):
        result = reedm(2, 4)
        assert result["rate"] == pytest.approx(result["k"] / result["n"])

    def test_invalid_r_gt_m(self):
        with pytest.raises(ValueError):
            reedm(4, 3)

    def test_invalid_m_zero(self):
        with pytest.raises(ValueError):
            reedm(0, 0)
