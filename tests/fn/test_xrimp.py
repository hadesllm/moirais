"""Tests for moirais.fn.xrimp -- SAR direct/indirect/total impacts"""

import numpy as np
import pytest

from moirais.fn.xrimp import sar_impacts


class TestSarImpacts:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sar_impacts(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = sar_impacts(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
