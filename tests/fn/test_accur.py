"""Tests for morie.fn.accur — classification accuracy."""
import numpy as np
import pytest
from morie.fn.accur import accuracy, accur


def test_perfect():
    r = accuracy([1, 0, 1, 0], [1, 0, 1, 0])
    assert abs(r.estimate - 1.0) < 1e-10


def test_zero():
    r = accuracy([1, 1, 1], [0, 0, 0])
    assert abs(r.estimate - 0.0) < 1e-10


def test_half():
    r = accuracy([1, 0, 1, 0], [1, 1, 0, 0])
    assert abs(r.estimate - 0.5) < 1e-10


def test_alias():
    assert accur is accuracy


def test_length_mismatch():
    with pytest.raises(ValueError):
        accuracy([1, 0], [1])


def test_empty():
    with pytest.raises(ValueError):
        accuracy([], [])


def test_extra_keys():
    r = accuracy([1, 0, 1, 0], [1, 0, 1, 0])
    assert r.extra["tp"] == 2
    assert r.extra["tn"] == 2
    assert r.extra["fp"] == 0
    assert r.extra["fn"] == 0
