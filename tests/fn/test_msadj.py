"""Test misadjustment (msadj)."""
from morie.fn.msadj import misadjustment, msadj
from morie.fn._containers import DescriptiveResult


class TestMsadj:
    def test_basic(self):
        result = misadjustment(0.01, 16, 1.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "misadjustment"
        assert abs(result.value - 0.08) < 1e-10

    def test_known_value(self):
        result = misadjustment(0.1, 10, 2.0)
        assert abs(result.value - 1.0) < 1e-10

    def test_alias(self):
        assert msadj is misadjustment
