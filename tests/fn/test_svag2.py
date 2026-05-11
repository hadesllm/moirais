"""Tests for morie.fn.svag2 -- 2D amendment agenda"""

import numpy as np
import pytest

from morie.fn.svag2 import agenda_2d


class TestAgenda2d:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = agenda_2d(data)
        assert result.value is not None

    def test_output_type(self):
        result = agenda_2d(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
