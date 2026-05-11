"""Tests for morie.fn.zsrch -- Random chi-squared field"""

import numpy as np
import pytest

from morie.fn.zsrch import random_chisq_field


class TestRandomChisqField:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = random_chisq_field(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = random_chisq_field(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
