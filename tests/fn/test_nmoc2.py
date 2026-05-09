"""Tests for moirais.fn.nmoc2 -- Optimal Classification 2D"""

import numpy as np
import pytest

from moirais.fn.nmoc2 import oc_2d


class TestOc2d:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = oc_2d(data)
        assert result.value is not None

    def test_output_type(self):
        result = oc_2d(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
