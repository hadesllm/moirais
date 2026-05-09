"""Tests for moirais.fn.zxcpv -- Vine copula spatial"""

import numpy as np
import pytest

from moirais.fn.zxcpv import copula_vine_sp


class TestCopulaVineSp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = copula_vine_sp(data)
        assert result.value is not None

    def test_output_type(self):
        result = copula_vine_sp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
