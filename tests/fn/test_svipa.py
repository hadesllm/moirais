"""Tests for morie.fn.svipa -- Adaptive ideal point estimation"""

import numpy as np
import pytest

from morie.fn.svipa import ideal_point_adapt


class TestIdealPointAdapt:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ideal_point_adapt(data)
        assert result.value is not None

    def test_output_type(self):
        result = ideal_point_adapt(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
