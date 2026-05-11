"""Tests for morie.fn.svagn -- 1D agenda setting equilibrium"""

import numpy as np
import pytest

from morie.fn.svagn import agenda_1d


class TestAgenda1d:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = agenda_1d(data)
        assert result.value is not None

    def test_output_type(self):
        result = agenda_1d(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
