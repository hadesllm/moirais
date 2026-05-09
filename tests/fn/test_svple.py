"""Tests for moirais.fn.svple -- Esteban-Ray polarization index"""

import numpy as np
import pytest

from moirais.fn.svple import polarization_er


class TestPolarizationEr:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = polarization_er(data)
        assert result.value is not None

    def test_output_type(self):
        result = polarization_er(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
