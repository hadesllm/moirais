"""Tests for morie.fn.cmprk — competing risks CIF."""
import numpy as np
import pytest
from morie.fn.cmprk import competing_risks


class TestCompetingRisks:
    def test_basic_cif(self):
        rng = np.random.default_rng(42)
        n = 200
        time = rng.exponential(2.0, size=n)
        event = rng.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])
        res = competing_risks(time, event, cause_of_interest=1)
        assert 0.0 <= res.extra["final_cif"] <= 1.0
        times = res.extra["times"]
        assert times == sorted(times)

    def test_cif_nondecreasing(self):
        rng = np.random.default_rng(42)
        n = 150
        time = rng.exponential(1.5, size=n)
        event = rng.choice([0, 1, 2], size=n, p=[0.2, 0.5, 0.3])
        res = competing_risks(time, event, cause_of_interest=1)
        cif = res.extra["cif"]
        for i in range(1, len(cif)):
            assert cif[i] >= cif[i - 1] - 1e-12
