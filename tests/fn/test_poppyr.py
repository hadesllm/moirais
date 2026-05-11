"""Tests for morie.fn.poppyr -- population pyramid."""

import numpy as np
from morie.fn.poppyr import population_pyramid


class TestPopulationPyramid:
    def test_basic(self):
        rng = np.random.default_rng(42)
        ages = rng.integers(0, 80, 200)
        sexes = rng.choice(["M", "F"], 200)
        res = population_pyramid(ages, sexes)
        assert res.name == "population_pyramid"
        assert res.extra["n"] == 200

    def test_custom_bins(self):
        res = population_pyramid([10, 20, 30], ["M", "F", "M"], bins=[0, 15, 25, 35])
        assert res.extra["n_bins"] == 3
