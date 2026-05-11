"""Tests for morie.fn.mtoped — pedestrian."""

import pytest
import pandas as pd
from morie.fn.mtoped import mto_pedestrian
from morie.fn._containers import DescriptiveResult


class TestPedestrian:
    def test_basic(self):
        df = pd.DataFrame({"pedestrian_involved": [1, 0, 1, 0, 1], "severity": ["Fatal", "Minor", "Serious", "Minor", "Minor"]})
        r = mto_pedestrian(df)
        assert isinstance(r, DescriptiveResult)
        assert r.value == pytest.approx(3.0)

    def test_severity_dist(self):
        df = pd.DataFrame({"pedestrian_involved": [1, 1], "severity": ["Fatal", "Minor"]})
        r = mto_pedestrian(df)
        assert r.extra["severity_dist"]["Fatal"] == 1
