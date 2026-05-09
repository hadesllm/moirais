"""Tests for moirais.fn.zxcpc -- Clayton copula spatial"""

import numpy as np
import pytest

from moirais.fn.zxcpc import copula_clayton_sp


class TestCopulaClaytonSp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = copula_clayton_sp(data)
        assert result.value is not None

    def test_output_type(self):
        result = copula_clayton_sp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
