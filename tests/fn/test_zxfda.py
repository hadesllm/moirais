"""Tests for morie.fn.zxfda -- Functional data analysis spatial"""

import numpy as np
import pytest

from morie.fn.zxfda import fda_spatial


class TestFdaSpatial:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = fda_spatial(data)
        assert result.value is not None

    def test_output_type(self):
        result = fda_spatial(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
