"""Tests for moirais.fn.ptlgc -- Log-Gaussian Cox process"""

import numpy as np
import pytest

from moirais.fn.ptlgc import log_gaussian_cox


class TestLogGaussianCox:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = log_gaussian_cox(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = log_gaussian_cox(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
