"""Tests for morie.fn.colsm -- stress-strain curve analysis."""

import numpy as np
from morie.fn.colsm import stress_strain, colsm
from morie.fn._containers import DescriptiveResult


class TestColsm:
    def test_alias(self):
        assert colsm is stress_strain

    def test_linear_material(self):
        eps = np.linspace(0, 0.01, 50)
        sig = 200e3 * eps
        r = stress_strain(eps, sig)
        assert isinstance(r, DescriptiveResult)
        assert r.value["youngs_modulus"] > 0
        assert r.value["uts"] > 0

    def test_toughness_positive(self):
        eps = np.linspace(0, 0.05, 100)
        sig = 200e3 * eps * np.exp(-10 * eps)
        r = stress_strain(eps, sig)
        assert r.value["toughness"] > 0
