"""Tests for morie.fn.ptkda -- Adaptive kernel density"""

import numpy as np
import pytest

from morie.fn.ptkda import kde_adaptive


class TestKdeAdaptive:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = kde_adaptive(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = kde_adaptive(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
