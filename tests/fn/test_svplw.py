"""Tests for moirais.fn.svplw -- Wolfson bipolarization index"""

import numpy as np
import pytest

from moirais.fn.svplw import polarization_wolf


class TestPolarizationWolf:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = polarization_wolf(data)
        assert result.value is not None

    def test_output_type(self):
        result = polarization_wolf(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
