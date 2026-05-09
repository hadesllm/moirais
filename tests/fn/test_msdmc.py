"""Tests for moirais.fn.msdmc -- Mahalanobis distance matrix"""

import numpy as np
import pytest

from moirais.fn.msdmc import dist_mahal


class TestDistMahal:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dist_mahal(data)
        assert result.value is not None

    def test_output_type(self):
        result = dist_mahal(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
