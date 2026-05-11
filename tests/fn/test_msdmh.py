"""Tests for morie.fn.msdmh -- Manhattan distance matrix"""

import numpy as np
import pytest

from morie.fn.msdmh import dist_manhattan


class TestDistManhattan:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dist_manhattan(data)
        assert result.value is not None

    def test_output_type(self):
        result = dist_manhattan(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
