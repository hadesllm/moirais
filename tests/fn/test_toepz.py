"""Tests for toeplitz_matrix."""
import numpy as np
from morie.fn.toepz import toeplitz_matrix, toepz


def test_basic():
    r = toeplitz_matrix([1, 2, 3])
    assert r.extra["shape"] == [3, 3]
    assert r.extra["symmetric"] is True


def test_alias():
    assert toepz is toeplitz_matrix


def test_non_symmetric():
    r = toeplitz_matrix([1, 2, 3], [1, 4, 5])
    assert r.extra["shape"] == [3, 3]
