"""Tests for moirais.fn.cyclo -- Gaussian beam optics."""

import numpy as np
from moirais.fn.cyclo import gaussian_beam, cyclo
from moirais.fn._containers import DescriptiveResult


class TestCyclo:
    def test_alias(self):
        assert cyclo is gaussian_beam

    def test_waist(self):
        r = gaussian_beam(wavelength=633e-9, w0=1e-3, z=[0])
        assert isinstance(r, DescriptiveResult)
        assert np.isclose(r.value["w"][0], 1e-3)

    def test_divergence(self):
        r = gaussian_beam(wavelength=633e-9, w0=1e-3, z=[0, 1, 2])
        assert r.value["w"][-1] > r.value["w"][0]
        assert r.value["z_rayleigh"] > 0
