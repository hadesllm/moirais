"""Tests for moirais.fn.svip2 -- 2D ideal point estimation"""

import numpy as np
import pytest

from moirais.fn.svip2 import ideal_point_2d


class TestIdealPoint2d:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ideal_point_2d(data)
        assert result.value is not None

    def test_output_type(self):
        result = ideal_point_2d(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
