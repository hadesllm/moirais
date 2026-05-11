"""Tests for morie.fn.vgani -- Anisotropy ratio estimation"""

import numpy as np
import pytest

from morie.fn.vgani import anisotropy_ratio


class TestAnisotropyRatio:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = anisotropy_ratio(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = anisotropy_ratio(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
