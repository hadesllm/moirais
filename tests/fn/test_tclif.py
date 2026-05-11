"""Tests for morie.fn.tclif — Clifford torus."""

import numpy as np
import pytest

from morie.fn.tclif import clifford_torus


class TestCliffordTorus:
    def test_on_s3(self):
        r = clifford_torus(n_points=100)
        norms = np.sqrt(
            r.extra["x1"]**2 + r.extra["x2"]**2 +
            r.extra["x3"]**2 + r.extra["x4"]**2
        )
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_flat(self):
        r = clifford_torus()
        assert r.extra["gaussian_curvature"] == 0.0

    def test_invalid(self):
        with pytest.raises(ValueError):
            clifford_torus(n_points=0)
