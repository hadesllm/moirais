"""Tests for morie.fn.svpl2 -- 2D spatial polarization"""

import numpy as np
import pytest

from morie.fn.svpl2 import polarization_2d


class TestPolarization2d:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = polarization_2d(data)
        assert result.value is not None

    def test_output_type(self):
        result = polarization_2d(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
