"""Test duty_cycle (sduty)."""
import numpy as np
from moirais.fn.sduty import duty_cycle, sduty
from moirais.fn._containers import DescriptiveResult


class TestDutyCycle:
    def test_all_above(self):
        x = np.array([1.0, 2.0, 3.0])
        result = duty_cycle(x, threshold=0.0)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 1.0) < 1e-10

    def test_half(self):
        x = np.array([-1.0, 1.0, -1.0, 1.0])
        assert abs(duty_cycle(x, threshold=0.0).value - 0.5) < 1e-10

    def test_alias(self):
        assert sduty is duty_cycle
