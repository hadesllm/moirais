"""Tests for moirais.fn.zecon -- Concentration index spatial"""

import numpy as np
import pytest

from moirais.fn.zecon import concentration_idx


class TestConcentrationIdx:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = concentration_idx(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = concentration_idx(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
