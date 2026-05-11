"""Tests for morie.fn.mxmod -- Semiparametric mixture model."""

import numpy as np
import pytest

from morie.fn.mxmod import mxmod


@pytest.fixture()
def mixture_data():
    rng = np.random.default_rng(42)
    c1 = rng.normal(0, 1, 150)
    c2 = rng.normal(5, 0.8, 100)
    return np.concatenate([c1, c2])


def test_returns_dict(mixture_data):
    result = mxmod(mixture_data, n_components=2)
    assert isinstance(result, dict)
    for k in ("means", "variances", "weights", "log_likelihood", "n_iter", "bic", "converged"):
        assert k in result


def test_two_components_means(mixture_data):
    result = mxmod(mixture_data, n_components=2)
    means = sorted(result["means"])
    assert means[0] < 3 < means[1]


def test_weights_sum_to_one(mixture_data):
    result = mxmod(mixture_data, n_components=2)
    assert sum(result["weights"]) == pytest.approx(1.0, abs=1e-6)


def test_converges(mixture_data):
    result = mxmod(mixture_data, n_components=2)
    assert result["converged"]


def test_bic_finite(mixture_data):
    result = mxmod(mixture_data, n_components=2)
    assert np.isfinite(result["bic"])


def test_single_component():
    data = np.random.default_rng(1).normal(0, 1, 50)
    result = mxmod(data, n_components=1)
    assert len(result["means"]) == 1
    assert result["weights"][0] == pytest.approx(1.0)


def test_empty_data():
    with pytest.raises(ValueError, match="empty"):
        mxmod(np.array([]))


def test_invalid_n_components():
    with pytest.raises(ValueError, match="n_components"):
        mxmod(np.array([1.0, 2.0]), n_components=0)


def test_cheatsheet():
    from morie.fn.mxmod import cheatsheet
    assert "mixture" in cheatsheet().lower()
