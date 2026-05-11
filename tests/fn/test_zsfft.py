"""Tests for morie.fn.zsfft -- FFT-based spectral simulation"""

import numpy as np
import pytest

from morie.fn.zsfft import spectral_sim


class TestSpectralSim:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spectral_sim(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = spectral_sim(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
