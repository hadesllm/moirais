"""Tests for moirais.fn.zepol -- Pollution surface estimation"""

import numpy as np
import pytest

from moirais.fn.zepol import pollution_surface


class TestPollutionSurface:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pollution_surface(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = pollution_surface(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
