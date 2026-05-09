"""Tests for moirais.fn.vgrdr -- Rodogram estimator"""

import numpy as np
import pytest

from moirais.fn.vgrdr import rodogram


class TestRodogram:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rodogram(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = rodogram(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
