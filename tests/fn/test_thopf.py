"""Tests for moirais.fn.thopf — Hopf fibration."""

import numpy as np
import pytest

from moirais.fn.thopf import hopf_fibration


class TestHopfFibration:
    def test_on_s2(self):
        r = hopf_fibration(n_points=100)
        norms = np.sqrt(r.extra["s2_x"]**2 + r.extra["s2_y"]**2 + r.extra["s2_z"]**2)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_s3_points(self):
        r = hopf_fibration(n_points=50)
        assert r.extra["s3"].shape == (50, 4)

    def test_invalid(self):
        with pytest.raises(ValueError):
            hopf_fibration(n_points=0)
