"""Tests for moirais.fn.svcls -- Minimum winning coalition size"""

import numpy as np
import pytest

from moirais.fn.svcls import coalition_size


class TestCoalitionSize:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = coalition_size(data)
        assert result.value is not None

    def test_output_type(self):
        result = coalition_size(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
