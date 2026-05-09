"""Tests for moirais.fn.zsidw -- IDW interpolation"""

import numpy as np
import pytest

from moirais.fn.zsidw import idw_interp


class TestIdwInterp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = idw_interp(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = idw_interp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
