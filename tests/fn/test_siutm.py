"""Tests for morie.fn.siutm — SIU timeline."""

import pytest
import pandas as pd
from morie.fn.siutm import siu_timeline
from morie.fn._containers import DescriptiveResult


class TestSiuTimeline:
    def test_basic(self):
        starts = pd.date_range("2023-01-01", periods=10, freq="ME")
        ends = starts + pd.Timedelta(days=90)
        r = siu_timeline(starts, ends)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["median_days"] == pytest.approx(90.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            siu_timeline([], [])
