"""Tests for moirais.fn.zsste -- Cressie-Huang space-time covariance"""

import numpy as np
import pytest

from moirais.fn.zsste import st_cressie_huang


class TestStCressieHuang:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = st_cressie_huang(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = st_cressie_huang(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
