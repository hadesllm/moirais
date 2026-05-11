"""Tests for morie.fn.svmpn -- Multi-party Nash equilibrium"""

import numpy as np
import pytest

from morie.fn.svmpn import multiparty_nash


class TestMultipartyNash:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = multiparty_nash(data)
        assert result.value is not None

    def test_output_type(self):
        result = multiparty_nash(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
