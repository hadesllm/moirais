"""Tests for morie.fn.svsep -- Separating hyperplane"""

import numpy as np
import pytest

from morie.fn.svsep import separating_hyp


class TestSeparatingHyp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = separating_hyp(data)
        assert result.value is not None

    def test_output_type(self):
        result = separating_hyp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
