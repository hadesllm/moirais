"""Tests for moirais.fn.zsstn -- Non-separable space-time covariance"""

import numpy as np
import pytest

from moirais.fn.zsstn import st_cov_nonsep


class TestStCovNonsep:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = st_cov_nonsep(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = st_cov_nonsep(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
