"""Tests for svd_decompose."""
import numpy as np
import pytest
from moirais.fn.svddp import svd_decompose, svddp


def test_identity():
    r = svd_decompose(np.eye(3))
    assert abs(r.value - 1.0) < 1e-10
    assert r.extra["rank"] == 3


def test_alias():
    assert svddp is svd_decompose


def test_rank_deficient():
    A = np.array([[1, 2], [2, 4]])
    r = svd_decompose(A)
    assert r.extra["rank"] == 1


def test_not_2d():
    with pytest.raises(ValueError):
        svd_decompose(np.array([1, 2, 3]))
