"""Tests for moirais.fn.ptisg -- Isotropic edge correction"""

import numpy as np
import pytest

from moirais.fn.ptisg import isotropic_guard


class TestIsotropicGuard:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = isotropic_guard(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = isotropic_guard(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
