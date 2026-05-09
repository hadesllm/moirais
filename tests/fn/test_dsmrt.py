"""Tests for moirais.fn.dsmrt -- Direct standardization."""

import pytest
from moirais.fn.dsmrt import direct_standardization


class TestDirectStandardization:
    def test_known(self):
        deaths = [10, 20, 30]
        pops = [1000, 2000, 3000]
        std = [0.3, 0.4, 0.3]
        res = direct_standardization(deaths, pops, std)
        assert res.measure == "direct_std_rate"
        assert res.estimate > 0

    def test_ci(self):
        res = direct_standardization([10, 20], [1000, 2000], [0.5, 0.5])
        assert res.ci_lower < res.estimate < res.ci_upper

    def test_mismatch(self):
        with pytest.raises(ValueError):
            direct_standardization([10], [1000, 2000], [0.5])
