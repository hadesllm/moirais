"""Tests for morie.fn.nmdmp -- Party divergence measure"""

import numpy as np
import pytest

from morie.fn.nmdmp import party_diverge


class TestPartyDiverge:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = party_diverge(data)
        assert result.value is not None

    def test_output_type(self):
        result = party_diverge(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
