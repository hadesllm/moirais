"""Tests for lda_features."""
import numpy as np
from moirais.fn.ldafd import lda_features, ldafd


def test_basic():
    rng = np.random.default_rng(42)
    X = np.vstack([rng.normal(0, 1, (50, 3)),
                   rng.normal(3, 1, (50, 3))])
    y = np.array([0]*50 + [1]*50)
    r = lda_features(X, y, n_components=1)
    assert r.extra["n_components"] == 1
    assert r.extra["n_classes"] == 2


def test_alias():
    assert ldafd is lda_features
