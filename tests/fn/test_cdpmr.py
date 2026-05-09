"""Tests for moirais.fn.cdpmr -- proportionate mortality."""

import pytest
from moirais.fn.cdpmr import proportionate_mortality


class TestProportionateMortality:
    def test_basic(self):
        res = proportionate_mortality(n_deaths_cause=200, total_deaths=1000)
        assert res.estimate == pytest.approx(0.2)

    def test_ci(self):
        res = proportionate_mortality(200, 1000)
        assert res.ci_lower < res.estimate < res.ci_upper
