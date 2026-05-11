"""Tests for morie.fn.nmpol -- Legislative polarity detection"""

import numpy as np
import pytest

from morie.fn.nmpol import leg_polarity


class TestLegPolarity:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = leg_polarity(data)
        assert result.value is not None

    def test_output_type(self):
        result = leg_polarity(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
