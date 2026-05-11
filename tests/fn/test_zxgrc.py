"""Tests for morie.fn.zxgrc -- Great circle distance"""

import numpy as np
import pytest

from morie.fn.zxgrc import great_circle


class TestGreatCircle:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = great_circle(data)
        assert result.value is not None

    def test_output_type(self):
        result = great_circle(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
