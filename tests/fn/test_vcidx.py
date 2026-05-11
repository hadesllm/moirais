"""Tests for morie.fn.vcidx — VC index / VC dimension computation."""

import numpy as np
import pytest

from morie.fn.vcidx import vcidx, VCIndexResult


def test_returns_result_type():
    X = np.random.default_rng(42).standard_normal((50, 2))
    result = vcidx(X, seed=1)
    assert isinstance(result, VCIndexResult)


def test_vc_dim_2d():
    X = np.random.default_rng(42).standard_normal((50, 2))
    result = vcidx(X, seed=1)
    assert result.vc_dim == 3
    assert result.theoretical_bound == 3


def test_vc_dim_1d():
    X = np.random.default_rng(7).standard_normal((30, 1))
    result = vcidx(X, seed=1)
    assert result.vc_dim == 2
    assert result.theoretical_bound == 2


def test_vc_dim_3d():
    X = np.random.default_rng(99).standard_normal((100, 3))
    result = vcidx(X, seed=1, max_k=5)
    assert result.vc_dim == 4
    assert result.theoretical_bound == 4


def test_shatter_counts_populated():
    X = np.random.default_rng(42).standard_normal((20, 1))
    result = vcidx(X, seed=1)
    assert len(result.shatter_counts) > 0
    for k, (shattered, total) in result.shatter_counts.items():
        assert shattered >= 0
        assert total > 0


def test_n_points_and_d():
    X = np.random.default_rng(42).standard_normal((40, 5))
    result = vcidx(X, seed=1, max_k=3)
    assert result.n_points == 40
    assert result.d_features == 5


def test_too_few_rows():
    with pytest.raises(ValueError, match="at least 2"):
        vcidx(np.array([[1.0, 2.0]]))


def test_unknown_classifier():
    X = np.random.default_rng(42).standard_normal((10, 2))
    with pytest.raises(ValueError, match="Unknown classifier"):
        vcidx(X, classifier_type="rbf")
