"""Tests for morie.fn.msdcr -- Correlation distance matrix"""

import numpy as np
import pytest

from morie.fn.msdcr import dist_correlation


class TestDistCorrelation:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dist_correlation(data)
        assert result.value is not None

    def test_output_type(self):
        result = dist_correlation(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
