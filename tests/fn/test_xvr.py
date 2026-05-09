"""Tests for moirais.fn.xvr -- partial correlation matrix."""

import numpy as np
from moirais.fn.xvr import partial_corr_matrix, xvr
from moirais.fn._containers import DescriptiveResult


class TestXvr:
    def test_alias(self):
        assert xvr is partial_corr_matrix

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
