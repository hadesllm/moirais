"""Tests for moirais.fn.mtoij — injury severity."""

import pytest
from moirais.fn.mtoij import mto_injury_severity
from moirais.fn._containers import DescriptiveResult


class TestInjurySeverity:
    def test_basic(self):
        r = mto_injury_severity(["Fatal"] * 2 + ["Serious"] * 10 + ["Minor"] * 88)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["proportions"]["Fatal"] == pytest.approx(0.02)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mto_injury_severity([])
