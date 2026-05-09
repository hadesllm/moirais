"""Tests for moirais.fn.svcli -- Optimal cutting line"""

import numpy as np
import pytest

from moirais.fn.svcli import cut_line


class TestCutLine:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cut_line(data)
        assert result.value is not None

    def test_output_type(self):
        result = cut_line(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
