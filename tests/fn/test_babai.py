"""Test babai_cvp."""
import numpy as np
import pytest
from moirais.fn.babai import babai_cvp
from moirais.fn._containers import DescriptiveResult


class TestBabaiCvp:
    def test_basic(self):
        basis = np.eye(3)
        target = np.array([1.3, 2.7, 0.1])
        result = babai_cvp(basis=basis, target=target)
        assert isinstance(result, DescriptiveResult)

    def test_output_type(self):
        basis = np.eye(3)
        target = np.array([1.3, 2.7, 0.1])
        result = babai_cvp(basis=basis, target=target)
        assert "closest_vector" in result.extra

    def test_closest_vector_near_lattice_point(self):
        basis = np.eye(3)
        target = np.array([1.3, 2.7, 0.1])
        result = babai_cvp(basis=basis, target=target)
        cv = np.asarray(result.extra["closest_vector"])
        expected = np.array([1.0, 3.0, 0.0])
        np.testing.assert_allclose(cv, expected, atol=1.0)
