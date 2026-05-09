"""Tests for moirais.fn.mstri -- Triangle inequality check"""

import numpy as np
import pytest

from moirais.fn.mstri import triangle_ineq


class TestTriangleIneq:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = triangle_ineq(data)
        assert result.value is not None

    def test_output_type(self):
        result = triangle_ineq(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
