"""Tests for morie.fn.xrlsa -- Local Moran's I (LISA)"""

import numpy as np
import pytest

from morie.fn.xrlsa import lisa_local


class TestLisaLocal:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = lisa_local(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = lisa_local(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
