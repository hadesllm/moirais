"""Tests for morie.fn.zxtn3 -- Three-way spatial tensor"""

import numpy as np
import pytest

from morie.fn.zxtn3 import tensor_3way_sp


class TestTensor3waySp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = tensor_3way_sp(data)
        assert result.value is not None

    def test_output_type(self):
        result = tensor_3way_sp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
