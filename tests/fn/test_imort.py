"""Tests for morie.fn.imort -- infant mortality."""

import pytest
from morie.fn.imort import infant_mortality


class TestInfantMortality:
    def test_basic(self):
        res = infant_mortality(n_infant_deaths=5, n_live_births=1000)
        assert res.estimate == pytest.approx(5.0)

    def test_ci(self):
        res = infant_mortality(5, 1000)
        assert res.ci_lower < res.estimate < res.ci_upper
