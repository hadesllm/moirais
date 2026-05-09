"""Tests for moirais.fn.ptkfn -- Ripley's K-function"""

import numpy as np
import pytest

from moirais.fn.ptkfn import k_function


class TestKFunction:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = k_function(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = k_function(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
