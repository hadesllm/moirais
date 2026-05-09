"""Tests for moirais.fn.pastr."""

import numpy as np
import pytest
from moirais.fn.pastr import pastr


class TestPastr:
    def test_basic(self):
        result = pastr(np.array([[1, 2, 3], [2, 1, 3], [3, 1, 2]]))
        assert result is not None
        assert result.statistic is not None
        assert isinstance(result.statistic, float)

    def test_returns_spatial_result(self):
        result = pastr(np.array([[1, 2, 3], [2, 1, 3], [3, 1, 2]]))
        assert hasattr(result, "statistic")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = pastr(np.array([[1, 2, 3], [2, 1, 3], [3, 1, 2]]))
        assert np.isfinite(result.statistic)

    def test_name_string(self):
        result = pastr(np.array([[1, 2, 3], [2, 1, 3], [3, 1, 2]]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
