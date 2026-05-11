"""Tests for morie.fn.nblam -- cloud mass function."""

import numpy as np
from morie.fn.nblam import cloud_mass_function, nblam
from morie.fn._containers import DescriptiveResult


class TestNblam:
    def test_alias(self):
        assert nblam is cloud_mass_function

    def test_power_law(self):
        rng = np.random.default_rng(42)
        masses = rng.pareto(1.5, 200) + 1
        r = cloud_mass_function(masses, n_bins=15)
        assert isinstance(r, DescriptiveResult)
        assert r.value["alpha_fit"] < 0
        assert r.value["total_mass"] > 0

    def test_bin_centers(self):
        masses = np.logspace(0, 3, 100)
        r = cloud_mass_function(masses, n_bins=10)
        assert len(r.value["bin_centers"]) == 10
