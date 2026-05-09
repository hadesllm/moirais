"""Tests for moirais.fn.svblp -- Bliss point estimation"""

import numpy as np
import pytest

from moirais.fn.svblp import bliss_point


class TestBlissPoint:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bliss_point(data)
        assert result.value is not None

    def test_output_type(self):
        result = bliss_point(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
