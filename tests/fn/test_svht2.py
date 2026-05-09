"""Tests for moirais.fn.svht2 -- Two-party Hotelling spatial competition"""

import numpy as np
import pytest

from moirais.fn.svht2 import hotelling_2party


class TestHotelling2party:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = hotelling_2party(data)
        assert result.value is not None

    def test_output_type(self):
        result = hotelling_2party(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
