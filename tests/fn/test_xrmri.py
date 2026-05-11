"""Tests for morie.fn.xrmri -- Moran's I on regression residuals"""

import numpy as np
import pytest

from morie.fn.xrmri import moran_resid


class TestMoranResid:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = moran_resid(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = moran_resid(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
