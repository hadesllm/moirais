"""Test group_delay (grpdl)."""
import numpy as np
from moirais.fn.grpdl import group_delay, grpdl
from moirais.fn._containers import DescriptiveResult


class TestGroupDelay:
    def test_basic(self):
        b = [1.0, 1.0]
        a = [1.0]
        result = group_delay(b, a)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "group_delay"

    def test_has_delay(self):
        b = [1.0, 1.0]
        a = [1.0]
        result = group_delay(b, a)
        assert "delay" in result.extra

    def test_alias(self):
        assert grpdl is group_delay
