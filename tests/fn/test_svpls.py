"""Tests for moirais.fn.svpls -- Party sorting index (Levendusky)"""

import numpy as np
import pytest

from moirais.fn.svpls import party_sorting


class TestPartySorting:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = party_sorting(data)
        assert result.value is not None

    def test_output_type(self):
        result = party_sorting(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
