"""Tests for morie.fn.zxcpg -- Gaussian copula spatial"""

import numpy as np
import pytest

from morie.fn.zxcpg import copula_gauss_sp


class TestCopulaGaussSp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = copula_gauss_sp(data)
        assert result.value is not None

    def test_output_type(self):
        result = copula_gauss_sp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
