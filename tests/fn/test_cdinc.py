"""Tests for morie.fn.cdinc -- incidence rate."""

import pytest
from morie.fn.cdinc import incidence_rate


class TestIncidenceRate:
    def test_basic(self):
        res = incidence_rate(n_new_cases=50, person_time=10000)
        assert res.estimate == pytest.approx(0.005)

    def test_ci(self):
        res = incidence_rate(50, 10000)
        assert res.ci_lower < res.estimate < res.ci_upper

    def test_invalid(self):
        with pytest.raises(ValueError):
            incidence_rate(10, 0)
