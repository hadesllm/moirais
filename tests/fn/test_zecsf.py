"""Tests for morie.fn.zecsf -- Concentration surface estimation"""

import numpy as np
import pytest

from morie.fn.zecsf import concentration_srf


class TestConcentrationSrf:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = concentration_srf(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = concentration_srf(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
