"""Tests for euclidean_dist."""
import numpy as np
import pytest
from moirais.fn.eucd import euclidean_dist, eucd


def test_basic():
    r = euclidean_dist([0, 0], [3, 4])
    assert abs(r.estimate - 5.0) < 1e-10


def test_alias():
    assert eucd is euclidean_dist


def test_same():
    r = euclidean_dist([1, 2], [1, 2])
    assert r.estimate == 0.0


def test_length_mismatch():
    with pytest.raises(ValueError):
        euclidean_dist([1], [1, 2])
