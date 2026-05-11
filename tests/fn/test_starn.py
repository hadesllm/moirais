"""Tests for morie.fn.starn — Spatio-temporal autoregressive model."""

import numpy as np
import pytest

from morie.fn.starn import starn


class TestStarn:

    def test_output_keys(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 5))
        W = np.eye(5) * 0 + 0.2
        np.fill_diagonal(W, 0)
        W = W / W.sum(axis=1, keepdims=True)
        result = starn(data, W)
        assert result["coefficients"] is not None
        assert result["fitted"].shape == (9, 5)

    def test_residual_shape(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((8, 4))
        W = np.ones((4, 4)) * 0.25
        np.fill_diagonal(W, 0)
        W = W / W.sum(axis=1, keepdims=True)
        result = starn(data, W, order=2)
        assert result["residuals"].shape == (7, 4)

    def test_weight_shape_mismatch(self):
        with pytest.raises(ValueError, match="weights must be"):
            starn(np.ones((5, 3)), np.eye(4))
