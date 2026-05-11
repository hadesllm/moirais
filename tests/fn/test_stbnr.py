"""Test stability_margin (stbnr)."""
from morie.fn.stbnr import stability_margin, stbnr
from morie.fn._containers import DescriptiveResult


class TestStbnr:
    def test_stable(self):
        result = stability_margin([1.0, -0.5, 0.2])
        assert isinstance(result, DescriptiveResult)
        assert result.name == "stability_margin"
        assert result.extra["is_stable"] is True
        assert result.value > 0

    def test_alias(self):
        assert stbnr is stability_margin
