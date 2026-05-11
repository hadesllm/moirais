"""Tests for morie.fn.svbrd -- Borda count in spatial model"""

import numpy as np
import pytest

from morie.fn.svbrd import borda_spatial


class TestBordaSpatial:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = borda_spatial(data)
        assert result.value is not None

    def test_output_type(self):
        result = borda_spatial(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
