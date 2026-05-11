"""Tests for morie.fn.zxvms -- Spatial von Mises distribution"""

import numpy as np
import pytest

from morie.fn.zxvms import von_mises_sp


class TestVonMisesSp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = von_mises_sp(data)
        assert isinstance(result.value, float) and np.isfinite(result.value)
        assert result.value == pytest.approx(3.0, rel=1e-10)

    def test_output_type(self):
        result = von_mises_sp(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result.value, float)
        assert isinstance(result.name, str) and len(result.name) > 0
