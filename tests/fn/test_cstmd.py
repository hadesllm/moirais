"""Tests for moirais.fn.cstmd — custody medical rate."""

import pytest
import numpy as np
from moirais.fn.cstmd import custody_medical
from moirais.fn._containers import CrimeResult


class TestCustodyMedical:

    def test_returns_crime(self):
        visits = np.array([10, 20, 15])
        pd_arr = np.array([1000, 2000, 1500])
        result = custody_medical(visits, pd_arr)
        assert isinstance(result, CrimeResult)
        assert result.rate > 0

    def test_rate_per_1000(self):
        result = custody_medical(np.array([100]), np.array([10000]))
        assert result.rate == pytest.approx(10.0)
