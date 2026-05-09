"""Tests for moirais.fn.svcle -- Coalition equilibrium (Schofield)"""

import numpy as np
import pytest

from moirais.fn.svcle import coalition_equil


class TestCoalitionEquil:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = coalition_equil(data)
        assert result.value is not None

    def test_output_type(self):
        result = coalition_equil(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
