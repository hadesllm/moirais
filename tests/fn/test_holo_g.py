"""Tests for moirais.fn.holo_g -- Geographic summary."""

import numpy as np
import pytest
from moirais.fn.holo_g import geo_summary, holo_g
from moirais.fn._containers import DescriptiveResult


class TestHoloG:
    def test_alias(self):
        assert holo_g is geo_summary

    def test_basic(self):
        values = np.array([10.0, 20.0, 30.0, 15.0, 25.0])
        regions = ["ON", "QC", "BC", "ON", "QC"]
        result = geo_summary(values, regions)
        assert isinstance(result, DescriptiveResult)
        assert result.value == 3
        stats = result.extra["region_stats"]
        assert stats["ON"]["mean"] == pytest.approx(12.5)
        assert stats["QC"]["mean"] == pytest.approx(22.5)
        assert stats["BC"]["n"] == 1

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            geo_summary(np.array([1, 2]), ["A"])
