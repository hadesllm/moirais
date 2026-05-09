"""Tests for moirais.fn.acmrt -- Age-cause-specific mortality."""

import pytest
from moirais.fn.acmrt import age_cause_mortality


class TestAgeCauseMortality:
    def test_basic(self):
        deaths = [[10, 5], [20, 10]]
        pops = [1000, 2000]
        res = age_cause_mortality(deaths, pops)
        assert res.measure == "age_cause_mortality"
        assert res.estimate > 0

    def test_per(self):
        res = age_cause_mortality(deaths=[[10]], populations=[100000], per=100000)
        assert res.estimate == pytest.approx(10.0)

    def test_mismatch(self):
        with pytest.raises(ValueError):
            age_cause_mortality(deaths=[[10]], populations=[100, 200])
