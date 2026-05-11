"""Tests for morie.fn.svmpc -- Multi-party spatial competition"""

import numpy as np
import pytest

from morie.fn.svmpc import multiparty_comp


class TestMultipartyComp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = multiparty_comp(data)
        assert result.value is not None

    def test_output_type(self):
        result = multiparty_comp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
