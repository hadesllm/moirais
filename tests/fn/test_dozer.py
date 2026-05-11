"""Tests for morie.fn.dozer -- PID controller."""

import numpy as np
from morie.fn.dozer import pid_tune, dozer
from morie.fn._containers import DescriptiveResult


class TestDozer:
    def test_alias(self):
        assert dozer is pid_tune

    def test_step_response(self):
        process = np.zeros(100)
        result = pid_tune(1.0, process, kp=0.5, ki=0.01, kd=0.01)
        assert isinstance(result, DescriptiveResult)
        assert result.value >= 0

    def test_already_at_setpoint(self):
        process = np.ones(50) * 5.0
        result = pid_tune(5.0, process, kp=1.0, ki=0.1, kd=0.05)
        assert result.value < 0.01
