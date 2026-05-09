"""Nature does not hurry, yet everything is accomplished. — Lao Tzu"""

import numpy as np
from moirais.fn.zionm import defense_allocation, zionm
from moirais.fn._containers import DescriptiveResult


class TestZionm:
    def test_alias(self):
        assert zionm is defense_allocation

    def test_proportional(self):
        result = defense_allocation([10, 20, 30], resources=100)
        assert isinstance(result, DescriptiveResult)
        alloc = result.value
        assert abs(sum(alloc) - 100) < 0.01

    def test_equal_threats(self):
        result = defense_allocation([5, 5, 5, 5], resources=100)
        for a in result.value:
            assert abs(a - 25.0) < 0.01
