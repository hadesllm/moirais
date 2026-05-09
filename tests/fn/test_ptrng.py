"""Tests for moirais.fn.ptrng -- Point pattern intensity"""

import numpy as np
import pytest

from moirais.fn.ptrng import pp_intensity


class TestPpIntensity:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pp_intensity(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = pp_intensity(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
