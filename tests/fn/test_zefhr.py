"""Tests for morie.fn.zefhr -- Fay-Herriot small area estimator"""

import numpy as np
import pytest

from morie.fn.zefhr import fay_herriot


class TestFayHerriot:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = fay_herriot(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = fay_herriot(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
