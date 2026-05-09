"""Tests for moirais.fn.grind -- Rosin-Rammler particle size distribution."""

import numpy as np
from moirais.fn.grind import rosin_rammler, grind
from moirais.fn._containers import DescriptiveResult


class TestGrind:
    def test_alias(self):
        assert grind is rosin_rammler

    def test_known_fit(self):
        d = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        d_star_true, n_true = 3.0, 1.5
        cp = 1 - np.exp(-(d / d_star_true) ** n_true)
        r = rosin_rammler(d, cp)
        assert isinstance(r, DescriptiveResult)
        assert abs(r.value["n"] - n_true) < 0.5

    def test_r_squared(self):
        d = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        cp = 1 - np.exp(-(d / 3.0) ** 1.5)
        r = rosin_rammler(d, cp)
        assert r.extra["r_squared"] > 0.9
