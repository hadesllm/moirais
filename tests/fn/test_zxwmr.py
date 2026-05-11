"""Tests for morie.fn.zxwmr -- Spatial wavelet MRA"""

import numpy as np
import pytest

from morie.fn.zxwmr import wavelet_mra_sp


class TestWaveletMraSp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wavelet_mra_sp(data)
        assert result.value is not None

    def test_output_type(self):
        result = wavelet_mra_sp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
