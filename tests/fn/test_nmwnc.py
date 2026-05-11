"""Tests for morie.fn.nmwnc -- W-NOMINATE classification"""

import numpy as np
import pytest

from morie.fn.nmwnc import wnominate_class


class TestWnominateClass:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wnominate_class(data)
        assert result.value is not None

    def test_output_type(self):
        result = wnominate_class(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
