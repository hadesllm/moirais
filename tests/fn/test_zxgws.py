"""Tests for moirais.fn.zxgws -- Geographically weighted summary stats"""

import numpy as np
import pytest

from moirais.fn.zxgws import gw_summary


class TestGwSummary:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gw_summary(data)
        assert result.value is not None

    def test_output_type(self):
        result = gw_summary(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
