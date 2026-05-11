"""Tests for morie.fn.zxtda -- Persistent homology spatial"""

import numpy as np
import pytest

from morie.fn.zxtda import tda_persistent


class TestTdaPersistent:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = tda_persistent(data)
        assert result.value is not None

    def test_output_type(self):
        result = tda_persistent(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
