"""Tests for morie.fn.zeear -- Ecological regression (Poisson)"""

import numpy as np
import pytest

from morie.fn.zeear import ecological_reg


class TestEcologicalReg:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ecological_reg(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = ecological_reg(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
