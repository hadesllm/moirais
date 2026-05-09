"""Tests for moirais.fn.rogue -- transfer learning feature extraction."""

import numpy as np
from moirais.fn.rogue import absorption_features, rogue
from moirais.fn._containers import DescriptiveResult


class TestRogue:
    def test_alias(self):
        assert rogue is absorption_features

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
