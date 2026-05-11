"""Tests for pcasg -- PCA signal decomposition."""
import numpy as np
import pytest
from morie.fn.pcasg import pcasg
from morie.fn._containers import DescriptiveResult


def test_pcasg_basic():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 100))
    result = pcasg(X)
    assert isinstance(result, DescriptiveResult)
    assert result.extra["components"].shape[0] == 3


def test_pcasg_explained_variance():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((4, 200))
    result = pcasg(X, n_components=2)
    ev = result.extra["explained_variance_ratio"]
    assert len(ev) == 2
    assert np.all(ev >= 0)


def test_pcasg_1d_error():
    with pytest.raises(ValueError):
        pcasg(np.array([1.0, 2.0, 3.0]))
