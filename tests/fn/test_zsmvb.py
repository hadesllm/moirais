"""Tests for moirais.fn.zsmvb -- Moving block bootstrap spatial"""

import numpy as np
import pytest

from moirais.fn.zsmvb import moving_block_boot


class TestMovingBlockBoot:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = moving_block_boot(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = moving_block_boot(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
