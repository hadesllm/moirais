"""Tests for morie.fn.ypll -- years of potential life lost."""

import numpy as np
import pytest
from morie.fn.ypll import years_potential_life_lost


class TestYPLL:
    def test_known(self):
        """Deaths at 30, 50, 80 with cutoff 75 => YPLL = 45+25+0 = 70."""
        ages = np.array([30, 50, 80])
        res = years_potential_life_lost(ages, cutoff=75)
        assert res.measure == "YPLL"
        assert res.estimate == pytest.approx(70.0)

    def test_all_above_cutoff(self):
        """All deaths above cutoff => YPLL = 0."""
        ages = np.array([80, 85, 90])
        res = years_potential_life_lost(ages, cutoff=75)
        assert res.estimate == pytest.approx(0.0)

    def test_n_premature(self):
        """2 of 3 deaths are premature."""
        ages = np.array([30, 50, 80])
        res = years_potential_life_lost(ages, cutoff=75)
        assert res.extra["n_premature"] == 2
