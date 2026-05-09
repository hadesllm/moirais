"""Tests for moirais.fn.zxutm -- UTM coordinate conversion"""

import numpy as np
import pytest

from moirais.fn.zxutm import utm_convert


class TestUtmConvert:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = utm_convert(data)
        assert result.value is not None

    def test_output_type(self):
        result = utm_convert(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
