"""Tests for moirais.fn.hadpr -- Hadamard product."""

import numpy as np
import pytest
from moirais.fn.hadpr import hadamard_product, hadpr
from moirais.fn._containers import DescriptiveResult


class TestHadpr:
    def test_alias(self):
        assert hadpr is hadamard_product

    def test_elementwise(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        B = np.array([[5, 6], [7, 8]], dtype=float)
        r = hadamard_product(A, B)
        assert isinstance(r, DescriptiveResult)
        np.testing.assert_allclose(r.extra["matrix"], [[5, 12], [21, 32]])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            hadamard_product(np.ones((2, 2)), np.ones((2, 3)))
