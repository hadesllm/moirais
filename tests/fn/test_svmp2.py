"""Tests for morie.fn.svmp2 -- Multi-party 2D equilibrium"""

import numpy as np
import pytest

from morie.fn.svmp2 import multiparty_2d


class TestMultiparty2d:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = multiparty_2d(data)
        assert result.value is not None

    def test_output_type(self):
        result = multiparty_2d(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
