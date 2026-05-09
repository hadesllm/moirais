"""Tests for moirais.fn.zsstc -- Separable space-time covariance"""

import numpy as np
import pytest

from moirais.fn.zsstc import st_cov_sep


class TestStCovSep:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = st_cov_sep(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = st_cov_sep(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
