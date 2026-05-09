"""Tests for moirais.fn.msink -- Three-way INDSCAL"""

import numpy as np
import pytest

from moirais.fn.msink import indscal_3way


class TestIndscal3way:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = indscal_3way(data)
        assert result.value is not None

    def test_output_type(self):
        result = indscal_3way(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
