"""Tests for moirais.fn.grwrt -- population growth rate."""

import pytest
from moirais.fn.grwrt import population_growth_rate


class TestPopGrowthRate:
    def test_doubling(self):
        import numpy as np
        res = population_growth_rate(pop_start=1000, pop_end=2000, years=10)
        assert res.estimate == pytest.approx(np.log(2) / 10)

    def test_decline(self):
        res = population_growth_rate(1000, 800, 5)
        assert res.estimate < 0
