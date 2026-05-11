"""Tests for morie.fn.zxlam -- Lambert conformal conic projection"""

import numpy as np
import pytest

from morie.fn.zxlam import lambert_proj


class TestLambertProj:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = lambert_proj(data)
        assert result.value is not None

    def test_output_type(self):
        result = lambert_proj(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
