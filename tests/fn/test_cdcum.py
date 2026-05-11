"""Tests for morie.fn.cdcum -- cumulative incidence."""

import pytest
from morie.fn.cdcum import cumulative_incidence


class TestCumulativeIncidence:
    def test_basic(self):
        res = cumulative_incidence(n_events=20, n_at_risk=200)
        assert res.estimate == pytest.approx(0.1)

    def test_ci(self):
        res = cumulative_incidence(20, 200)
        assert res.ci_lower < res.estimate < res.ci_upper

    def test_invalid(self):
        with pytest.raises(ValueError):
            cumulative_incidence(10, 0)
