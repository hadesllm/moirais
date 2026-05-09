"""Tests for moirais.fn.kgsmw -- Simple kriging weights"""

import numpy as np
import pytest

from moirais.fn.kgsmw import sk_weights


class TestSkWeights:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sk_weights(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = sk_weights(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
