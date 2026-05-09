"""Tests for moirais.fn.svipm -- MLE ideal point estimation"""

import numpy as np
import pytest

from moirais.fn.svipm import ideal_point_mle


class TestIdealPointMle:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ideal_point_mle(data)
        assert result.value is not None

    def test_output_type(self):
        result = ideal_point_mle(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
