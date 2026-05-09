"""Tests for moirais.fn.msshp -- Shepard diagram values"""

import numpy as np
import pytest

from moirais.fn.msshp import shepard_diag


class TestShepardDiag:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = shepard_diag(data)
        assert result.value is not None

    def test_output_type(self):
        result = shepard_diag(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
