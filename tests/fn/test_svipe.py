"""Tests for morie.fn.svipe -- EM ideal point estimation"""

import numpy as np
import pytest

from morie.fn.svipe import ideal_point_em


class TestIdealPointEm:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ideal_point_em(data)
        assert result.value is not None

    def test_output_type(self):
        result = ideal_point_em(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
