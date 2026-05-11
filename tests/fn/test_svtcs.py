"""Tests for morie.fn.svtcs -- Top cycle set computation"""

import numpy as np
import pytest

from morie.fn.svtcs import top_cycle_set


class TestTopCycleSet:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = top_cycle_set(data)
        assert result.value is not None

    def test_output_type(self):
        result = top_cycle_set(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
