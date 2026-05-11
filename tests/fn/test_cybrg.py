"""Tests for morie.fn.cybrg -- PID control simulation."""

from morie.fn.cybrg import pid_simulate, cybrg
from morie.fn._containers import DescriptiveResult


class TestCybrg:
    def test_alias(self):
        assert cybrg is pid_simulate

    def test_settles(self):
        result = pid_simulate(1.0, kp=2.0, ki=0.5, kd=0.1, n_steps=1000)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.extra["final_output"] - 1.0) < 0.1

    def test_zero_setpoint(self):
        result = pid_simulate(0.0, n_steps=100)
        assert abs(result.extra["final_output"]) < 0.01
