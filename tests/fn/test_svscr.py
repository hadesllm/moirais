"""Tests for moirais.fn.svscr -- Scree test for spatial dimensions"""

import numpy as np
import pytest

from moirais.fn.svscr import scree_spatial


class TestScreeSpatial:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scree_spatial(data)
        assert result.value is not None

    def test_output_type(self):
        result = scree_spatial(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
