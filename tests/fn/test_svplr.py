"""Tests for moirais.fn.svplr -- 1D ideological polarization index"""

import numpy as np
import pytest

from moirais.fn.svplr import polarization_1d


class TestPolarization1d:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = polarization_1d(data)
        assert result.value is not None

    def test_output_type(self):
        result = polarization_1d(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
