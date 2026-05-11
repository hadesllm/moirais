"""Tests for morie.fn.sumort -- substance mortality."""

import pytest
from morie.fn.sumort import substance_mortality


class TestSubstanceMortality:
    def test_basic(self):
        res = substance_mortality(deaths_attributed=200, total_deaths=1000)
        assert res.estimate == pytest.approx(0.2)

    def test_ci(self):
        res = substance_mortality(200, 1000)
        assert res.ci_lower < res.estimate < res.ci_upper

    def test_invalid(self):
        with pytest.raises(ValueError):
            substance_mortality(100, 0)
