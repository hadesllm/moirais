"""Tests for moirais.fn.svclv -- Coalition value in spatial game"""

import numpy as np
import pytest

from moirais.fn.svclv import coalition_value


class TestCoalitionValue:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = coalition_value(data)
        assert result.value is not None

    def test_output_type(self):
        result = coalition_value(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
