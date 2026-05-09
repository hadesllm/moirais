"""Tests for moirais.fn.kgunw -- Universal kriging weights"""

import numpy as np
import pytest

from moirais.fn.kgunw import uk_weights


class TestUkWeights:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = uk_weights(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = uk_weights(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
