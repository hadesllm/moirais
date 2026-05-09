"""Tests for moirais.fn.msdcs -- Cosine distance matrix"""

import numpy as np
import pytest

from moirais.fn.msdcs import dist_cosine


class TestDistCosine:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dist_cosine(data)
        assert result.value is not None

    def test_output_type(self):
        result = dist_cosine(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
