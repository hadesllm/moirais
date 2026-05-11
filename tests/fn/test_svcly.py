"""Tests for morie.fn.svcly -- Yolk of spatial game"""

import numpy as np
import pytest

from morie.fn.svcly import coalition_yolk


class TestCoalitionYolk:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = coalition_yolk(data)
        assert result.value is not None

    def test_output_type(self):
        result = coalition_yolk(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
