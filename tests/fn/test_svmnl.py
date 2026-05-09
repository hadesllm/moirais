"""Tests for moirais.fn.svmnl -- Multinomial spatial choice model"""

import numpy as np
import pytest

from moirais.fn.svmnl import multinomial_spatial


class TestMultinomialSpatial:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = multinomial_spatial(data)
        assert result.value is not None

    def test_output_type(self):
        result = multinomial_spatial(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
