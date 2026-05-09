"""Tests for moirais.fn.ihmrt -- indigenous mortality."""

import pytest
from moirais.fn.ihmrt import indigenous_mortality


class TestIndigenousMortality:
    def test_basic(self):
        res = indigenous_mortality(100, 500, 10000, 100000)
        assert res.estimate == pytest.approx(2.0)

    def test_ci(self):
        res = indigenous_mortality(100, 500, 10000, 100000)
        assert res.ci_lower < res.estimate < res.ci_upper
