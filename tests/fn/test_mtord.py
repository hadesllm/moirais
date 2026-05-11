"""Tests for morie.fn.mtord — road segment."""

import pytest
import pandas as pd
from morie.fn.mtord import mto_road_segment
from morie.fn._containers import DescriptiveResult


class TestRoadSegment:
    def test_basic(self):
        df = pd.DataFrame({"segment_id": ["S1", "S1", "S2"], "n_crashes": [5, 3, 10], "aadt": [10000, 10000, 20000]})
        r = mto_road_segment(df)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["n_segments"] == 2

    def test_missing_col(self):
        with pytest.raises(ValueError):
            mto_road_segment(pd.DataFrame({"x": [1]}))
