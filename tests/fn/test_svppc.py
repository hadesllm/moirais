"""Tests for morie.fn.svppc -- Congressional party position"""

import numpy as np
import pytest

from morie.fn.svppc import party_congress


class TestPartyCongress:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = party_congress(data)
        assert result.value is not None

    def test_output_type(self):
        result = party_congress(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
