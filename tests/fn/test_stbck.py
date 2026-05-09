"""Test stability_check (stbck)."""
import numpy as np
from moirais.fn.stbck import stability_check, stbck
from moirais.fn._containers import DescriptiveResult


class TestStabilityCheck:
    def test_stable(self):
        b = [1.0]
        a = [1.0, -0.5]
        result = stability_check(b, a)
        assert isinstance(result, DescriptiveResult)
        assert result.value == 1.0
        assert result.extra["stable"] is True

    def test_unstable(self):
        b = [1.0]
        a = [1.0, -1.5]
        result = stability_check(b, a)
        assert result.value == 0.0
        assert result.extra["stable"] is False

    def test_alias(self):
        assert stbck is stability_check
