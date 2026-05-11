"""Tests for morie.fn.zxgat -- Graph attention spatial"""

import numpy as np
import pytest

from morie.fn.zxgat import graph_attention_sp


class TestGraphAttentionSp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = graph_attention_sp(data)
        assert result.value is not None

    def test_output_type(self):
        result = graph_attention_sp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
