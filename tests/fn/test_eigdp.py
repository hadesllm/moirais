"""Tests for eigen_decompose."""
import numpy as np
import pytest
from morie.fn.eigdp import eigen_decompose, eigdp


def test_identity():
    r = eigen_decompose(np.eye(3))
    assert abs(r.value - 1.0) < 1e-10


def test_alias():
    assert eigdp is eigen_decompose


def test_not_square():
    with pytest.raises(ValueError):
        eigen_decompose(np.ones((2, 3)))


def test_symmetric():
    A = np.array([[2, 1], [1, 2]])
    r = eigen_decompose(A)
    assert r.extra["symmetric"] is True
    assert abs(r.extra["eigenvalues"][0] - 3.0) < 1e-10
