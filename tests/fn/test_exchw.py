"""Tests for morie.fn.exchw — exchangeable bootstrap weights."""

import numpy as np
import pytest

from morie.fn.exchw import exchw


def test_dirichlet_shape():
    result = exchw(50, n_boot=100, seed=42)
    assert result["weights"].shape == (100, 50)


def test_dirichlet_sum():
    result = exchw(20, n_boot=10, seed=7)
    row_sums = np.sum(result["weights"], axis=1)
    np.testing.assert_allclose(row_sums, 20.0, atol=1e-10)


def test_multinomial():
    result = exchw(30, n_boot=50, distribution="multinomial", seed=42)
    row_sums = np.sum(result["weights"], axis=1)
    np.testing.assert_allclose(row_sums, 30.0)


def test_invalid_n_raises():
    with pytest.raises(ValueError, match="n must be"):
        exchw(0)
