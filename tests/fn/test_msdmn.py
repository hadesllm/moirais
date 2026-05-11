"""Tests for morie.fn.msdmn -- Minkowski distance matrix"""

import numpy as np
import pytest

from morie.fn.msdmn import dist_mink


class TestDistMink:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dist_mink(data)
        assert result.value is not None

    def test_output_type(self):
        result = dist_mink(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
