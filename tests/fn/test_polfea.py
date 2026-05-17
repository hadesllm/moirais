"""Tests for morie.fn.polfea -- polynomial feature expansion."""

import numpy as np
from morie.fn.polfea import polynomial_features, polfea
from morie.fn._containers import DescriptiveResult


class TestPolfea:
    def test_alias(self):
        assert polfea is polynomial_features

    def test_degree2(self):
        X = np.array([[1, 2], [3, 4]], dtype=float)
        r = polynomial_features(X, degree=2)
        assert isinstance(r, DescriptiveResult)
        assert r.value.shape[0] == 2
        assert r.value.shape[1] == 5

    def test_interaction_only(self):
        X = np.array([[1, 2, 3]], dtype=float)
        r = polynomial_features(X, degree=2, interaction_only=True)
        assert r.value.shape[1] == 6
