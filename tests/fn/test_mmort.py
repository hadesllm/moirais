"""Tests for moirais.fn.mmort -- maternal mortality."""

import pytest
from moirais.fn.mmort import maternal_mortality


class TestMaternalMortality:
    def test_basic(self):
        res = maternal_mortality(n_maternal_deaths=10, n_live_births=100000)
        assert res.estimate == pytest.approx(10.0)

    def test_ci(self):
        res = maternal_mortality(10, 100000)
        assert res.ci_lower < res.estimate < res.ci_upper
