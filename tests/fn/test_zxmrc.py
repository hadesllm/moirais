"""Tests for morie.fn.zxmrc -- Mercator projection"""

import numpy as np
import pytest

from morie.fn.zxmrc import mercator_proj


class TestMercatorProj:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mercator_proj(data)
        assert result.value is not None

    def test_output_type(self):
        result = mercator_proj(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
