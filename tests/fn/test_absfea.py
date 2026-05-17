"""Tests for morie.fn.absfea -- transfer learning feature extraction."""

import numpy as np
from morie.fn.absfea import absorption_features, absfea
from morie.fn._containers import DescriptiveResult


class TestAbsfea:
    def test_alias(self):
        assert absfea is absorption_features

    def test_projection(self):
        rng = np.random.default_rng(42)
        S = rng.normal(0, 1, (50, 10))
        T = rng.normal(0, 1, (20, 10))
        r = absorption_features(S, T, n_components=3)
        assert isinstance(r, DescriptiveResult)
        assert r.value["projected"].shape == (20, 3)
        assert len(r.value["explained_variance_ratio"]) == 3

    def test_full_components(self):
        rng = np.random.default_rng(0)
        S = rng.normal(0, 1, (30, 5))
        T = rng.normal(0, 1, (10, 5))
        r = absorption_features(S, T)
        assert r.value["projected"].shape[1] == 5
