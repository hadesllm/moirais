"""Tests for moirais.fn.svplc -- Plott radial symmetry condition check"""

import numpy as np
import pytest

from moirais.fn.svplc import plott_condition


class TestPlottCondition:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = plott_condition(data)
        assert result.value is not None

    def test_output_type(self):
        result = plott_condition(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
