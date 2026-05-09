"""Test svp_approx."""
import numpy as np
import pytest
from moirais.fn.svpap import svp_approx
from moirais.fn._containers import DescriptiveResult


class TestSvpApprox:
    def test_basic(self):
        basis = np.array([[2, 1], [1, 3]], dtype=float)
        result = svp_approx(basis=basis)
        assert isinstance(result, DescriptiveResult)

    def test_output_type(self):
        basis = np.array([[2, 1], [1, 3]], dtype=float)
        result = svp_approx(basis=basis)
        assert "shortest_vector" in result.extra

    def test_norm_positive(self):
        basis = np.array([[2, 1], [1, 3]], dtype=float)
        result = svp_approx(basis=basis)
        sv = np.asarray(result.extra["shortest_vector"])
        assert np.linalg.norm(sv) > 0
