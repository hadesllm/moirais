"""Tests for moirais.fn.xrfgt -- Getis spatial filtering"""

import numpy as np
import pytest

from moirais.fn.xrfgt import getis_filter


class TestGetisFilter:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = getis_filter(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = getis_filter(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
