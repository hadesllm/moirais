"""Tests for moirais.fn.efart -- factor rotation."""

import numpy as np
from moirais.fn.efart import efa_rotate


class TestEfaRotate:

    def test_varimax_shape(self):
        rng = np.random.default_rng(42)
        loadings = rng.standard_normal((10, 3))
        result = efa_rotate(loadings, method="varimax")
        assert result.shape == (10, 3)

    def test_promax_shape(self):
        rng = np.random.default_rng(42)
        loadings = rng.standard_normal((10, 3))
        result = efa_rotate(loadings, method="promax")
        assert result.shape == (10, 3)

    def test_single_factor_unchanged(self):
        rng = np.random.default_rng(42)
        loadings = rng.standard_normal((5, 1))
        result = efa_rotate(loadings)
        np.testing.assert_allclose(result, loadings)
