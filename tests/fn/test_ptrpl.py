"""Tests for moirais.fn.ptrpl -- Ripley edge correction"""

import numpy as np
import pytest

from moirais.fn.ptrpl import ripley_correction


class TestRipleyCorrection:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ripley_correction(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = ripley_correction(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
