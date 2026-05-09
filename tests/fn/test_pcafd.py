"""Tests for pca_features."""
import numpy as np
import pytest
from moirais.fn.pcafd import pca_features, pcafd


def test_basic():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    r = pca_features(X, n_components=2)
    assert r.extra["n_components"] == 2
    assert len(r.extra["explained_variance_ratio"]) == 2


def test_alias():
    assert pcafd is pca_features


def test_too_few():
    with pytest.raises(ValueError):
        pca_features([[1, 2]])
