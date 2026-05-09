"""Tests for moirais.fn.zxgcn -- Graph convolution spatial"""

import numpy as np
import pytest

from moirais.fn.zxgcn import graph_conv_sp


class TestGraphConvSp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = graph_conv_sp(data)
        assert result.value is not None

    def test_output_type(self):
        result = graph_conv_sp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
