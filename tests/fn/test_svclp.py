"""Tests for moirais.fn.svclp -- Cutting plane in 3D"""

import numpy as np
import pytest

from moirais.fn.svclp import cut_plane


class TestCutPlane:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cut_plane(data)
        assert result.value is not None

    def test_output_type(self):
        result = cut_plane(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
