"""Tests for morie.fn.svbnk -- Banks set computation"""

import numpy as np
import pytest

from morie.fn.svbnk import banks_set


class TestBanksSet:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = banks_set(data)
        assert result.value is not None

    def test_output_type(self):
        result = banks_set(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
