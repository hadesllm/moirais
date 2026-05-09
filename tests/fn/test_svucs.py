"""Tests for moirais.fn.svucs -- Uncovered set in 2D"""

import numpy as np
import pytest

from moirais.fn.svucs import uncovered_set


class TestUncoveredSet:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = uncovered_set(data)
        assert result.value is not None

    def test_output_type(self):
        result = uncovered_set(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
