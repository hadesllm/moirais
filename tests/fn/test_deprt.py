"""Tests for moirais.fn.deprt -- dependency ratio."""

import pytest
from moirais.fn.deprt import dependency_ratio


class TestDependencyRatio:
    def test_basic(self):
        res = dependency_ratio(pop_0_14=200, pop_15_64=600, pop_65plus=200)
        assert res.estimate == pytest.approx(400 / 600 * 100)

    def test_zero_working(self):
        with pytest.raises(ValueError):
            dependency_ratio(100, 0, 50)
