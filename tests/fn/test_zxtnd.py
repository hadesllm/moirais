"""Tests for morie.fn.zxtnd -- Spatial tensor decomposition"""

import numpy as np
import pytest

from morie.fn.zxtnd import tensor_decomp_sp


class TestTensorDecompSp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = tensor_decomp_sp(data)
        assert result.value is not None

    def test_output_type(self):
        result = tensor_decomp_sp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
