"""Tests for icasg -- ICA source separation."""
import numpy as np
import pytest
from morie.fn.icasg import icasg
from morie.fn._containers import DescriptiveResult


def test_icasg_basic():
    rng = np.random.default_rng(42)
    s1 = np.sin(2 * np.pi * 5 * np.arange(500) / 500)
    s2 = rng.standard_normal(500)
    S = np.vstack([s1, s2])
    A = np.array([[1.0, 0.5], [0.3, 1.0]])
    X = A @ S
    result = icasg(X)
    assert isinstance(result, DescriptiveResult)
    assert result.extra["sources"].shape == (2, 500)


def test_icasg_mixing_matrix():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((3, 200))
    result = icasg(X, n_components=2)
    assert result.extra["mixing_matrix"].shape[1] == 2


def test_icasg_1d_error():
    with pytest.raises(ValueError):
        icasg(np.array([1.0, 2.0, 3.0]))
