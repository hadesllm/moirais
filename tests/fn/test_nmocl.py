"""Tests for moirais.fn.nmocl -- Optimal Classification cutting line"""

import numpy as np
import pytest

from moirais.fn.nmocl import oc_cutline


class TestOcCutline:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = oc_cutline(data)
        assert result.value is not None

    def test_output_type(self):
        result = oc_cutline(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
