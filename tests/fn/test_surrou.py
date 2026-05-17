"""Tests for morie.fn.surrou -- surface roughness."""

import numpy as np
from morie.fn.surrou import surface_roughness, surrou
from morie.fn._containers import DescriptiveResult


class TestSurrou:
    def test_alias(self):
        assert surrou is surface_roughness

    def test_flat_surface(self):
        profile = np.zeros(100)
        r = surface_roughness(profile)
        assert isinstance(r, DescriptiveResult)
        assert r.value == 0.0

    def test_sinusoidal(self):
        x = np.sin(np.linspace(0, 4 * np.pi, 200))
        r = surface_roughness(x)
        assert r.value > 0
        assert r.extra["Rz"] > 0
