"""Test aliasing_demo (alias)."""
import numpy as np
from moirais.fn.alias import aliasing_demo, alias
from moirais.fn._containers import DescriptiveResult


class TestAliasingDemo:
    def test_no_aliasing(self):
        result = aliasing_demo(f_signal=100.0, fs=500.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "aliasing_demo"
        assert result.extra["is_aliased"] is False

    def test_aliasing(self):
        result = aliasing_demo(f_signal=400.0, fs=500.0)
        assert result.extra["is_aliased"] is True
        assert abs(result.value - 100.0) < 1e-10

    def test_alias_ref(self):
        assert alias is aliasing_demo
