"""Tests for morie.fn.parmat -- partial correlation matrix."""

import numpy as np
from morie.fn.parmat import partial_corr_matrix, parmat
from morie.fn._containers import DescriptiveResult


class TestParmat:
    def test_alias(self):
        assert parmat is partial_corr_matrix

    def test_diagonal_ones(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 4))
        r = partial_corr_matrix(X)
        assert isinstance(r, DescriptiveResult)
        pcorr = r.value["partial_corr"]
        assert np.allclose(np.diag(pcorr), 1.0)

    def test_density(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (100, 3))
        r = partial_corr_matrix(X)
        assert 0 <= r.value["density"] <= 1
