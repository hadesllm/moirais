"""Tests for morie.fn.cdmrt -- cause-specific mortality."""

import pytest
from morie.fn.cdmrt import cause_specific_mortality


class TestCauseSpecificMortality:
    def test_basic(self):
        res = cause_specific_mortality(n_deaths_cause=50, population=100000)
        assert res.estimate == pytest.approx(50.0)

    def test_ci(self):
        res = cause_specific_mortality(50, 100000)
        assert res.ci_lower < res.estimate < res.ci_upper
