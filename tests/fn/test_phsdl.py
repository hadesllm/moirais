"""Test phase_delay (phsdl)."""
import numpy as np
from moirais.fn.phsdl import phase_delay, phsdl
from moirais.fn._containers import DescriptiveResult


class TestPhaseDelay:
    def test_basic(self):
        b = [1.0, 0.5]
        a = [1.0]
        result = phase_delay(b, a)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "phase_delay"

    def test_has_delay(self):
        b = [1.0, 0.5]
        a = [1.0]
        result = phase_delay(b, a)
        assert "delay" in result.extra

    def test_alias(self):
        assert phsdl is phase_delay
