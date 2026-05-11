"""Tests for morie.fn.svplf -- Affective polarization measure"""

import numpy as np
import pytest

from morie.fn.svplf import polarization_aff


class TestPolarizationAff:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = polarization_aff(data)
        assert result.value is not None

    def test_output_type(self):
        result = polarization_aff(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
