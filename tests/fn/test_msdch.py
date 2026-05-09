"""Tests for moirais.fn.msdch -- Chebyshev distance matrix"""

import numpy as np
import pytest

from moirais.fn.msdch import dist_chebyshev


class TestDistChebyshev:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dist_chebyshev(data)
        assert result.value is not None

    def test_output_type(self):
        result = dist_chebyshev(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
