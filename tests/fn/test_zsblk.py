"""Tests for morie.fn.zsblk -- Spatial block bootstrap"""

import numpy as np
import pytest

from morie.fn.zsblk import block_bootstrap


class TestBlockBootstrap:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = block_bootstrap(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = block_bootstrap(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
