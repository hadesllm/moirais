"""Tests for morie.fn.precn — precision (PPV)."""
import numpy as np
import pytest
from morie.fn.precn import precision, precn


def test_perfect():
    r = precision([1, 0, 1, 0], [1, 0, 1, 0])
    assert abs(r.estimate - 1.0) < 1e-10


def test_half_precision():
    r = precision([1, 0, 0, 0], [1, 1, 0, 0])
    assert abs(r.estimate - 0.5) < 1e-10


def test_no_positive_predictions():
    r = precision([1, 1], [0, 0])
    assert abs(r.estimate - 0.0) < 1e-10


def test_alias():
    assert precn is precision


def test_length_mismatch():
    with pytest.raises(ValueError):
        precision([1], [0, 1])


def test_extra_keys():
    r = precision([1, 0, 1, 0], [1, 1, 0, 0])
    assert r.extra["tp"] == 1
    assert r.extra["fp"] == 1
