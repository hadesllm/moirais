"""Tests for morie.fn.xrlsg -- Local Getis-Ord Gi*"""

import numpy as np
import pytest

from morie.fn.xrlsg import lisa_getis


class TestLisaGetis:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = lisa_getis(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = lisa_getis(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
