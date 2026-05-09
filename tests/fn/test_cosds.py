"""Tests for cosine_distance."""
import numpy as np
import pytest
from moirais.fn.cosds import cosine_distance, cosds


def test_identical():
    r = cosine_distance([1, 0], [1, 0])
    assert abs(r.estimate) < 1e-10


def test_orthogonal():
    r = cosine_distance([1, 0], [0, 1])
    assert abs(r.estimate - 1.0) < 1e-10


def test_alias():
    assert cosds is cosine_distance


def test_zero_vec():
    with pytest.raises(ValueError):
        cosine_distance([0, 0], [1, 0])
