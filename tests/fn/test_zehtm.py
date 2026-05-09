"""Tests for moirais.fn.zehtm -- Hotspot detection map"""

import numpy as np
import pytest

from moirais.fn.zehtm import hotspot_map


class TestHotspotMap:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = hotspot_map(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = hotspot_map(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
