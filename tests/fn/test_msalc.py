"""Tests for morie.fn.msalc -- Alienation coefficient"""

import numpy as np
import pytest

from morie.fn.msalc import alienation


class TestAlienation:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = alienation(data)
        assert result.value is not None

    def test_output_type(self):
        result = alienation(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
