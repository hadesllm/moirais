"""Tests for morie.fn.nmdwp -- DW-NOMINATE polarization"""

import numpy as np
import pytest

from morie.fn.nmdwp import dwnominate_polar


class TestDwnominatePolar:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dwnominate_polar(data)
        assert result.value is not None

    def test_output_type(self):
        result = dwnominate_polar(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
