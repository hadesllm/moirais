"""Tests for pseudo_inverse."""
import numpy as np
import pytest
from moirais.fn.psinv import pseudo_inverse, psinv


def test_identity():
    r = pseudo_inverse(np.eye(3))
    assert r.extra["rank"] == 3


def test_alias():
    assert psinv is pseudo_inverse


def test_non_square():
    A = np.array([[1, 2], [3, 4], [5, 6]])
    r = pseudo_inverse(A)
    assert r.extra["pinv_shape"] == [2, 3]


def test_not_2d():
    with pytest.raises(ValueError):
        pseudo_inverse(np.array([1, 2]))
