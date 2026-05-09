"""Tests for moirais.fn.svipn -- Normal ideal point model"""

import numpy as np
import pytest

from moirais.fn.svipn import ideal_point_normal


class TestIdealPointNormal:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ideal_point_normal(data)
        assert result.value is not None

    def test_output_type(self):
        result = ideal_point_normal(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
