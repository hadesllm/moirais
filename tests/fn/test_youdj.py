"""Tests for moirais.fn.youdj — Youden's J statistic."""
import numpy as np
import pytest
from moirais.fn.youdj import youdens_j, youdj


def test_perfect():
    r = youdens_j([1, 0, 1, 0], [1, 0, 1, 0])
    assert abs(r.estimate - 1.0) < 1e-10


def test_random():
    r = youdens_j([1, 1, 0, 0], [1, 0, 1, 0])
    assert abs(r.estimate - 0.0) < 1e-10


def test_inverse():
    r = youdens_j([1, 0, 1, 0], [0, 1, 0, 1])
    assert abs(r.estimate - (-1.0)) < 1e-10


def test_alias():
    assert youdj is youdens_j


def test_length_mismatch():
    with pytest.raises(ValueError):
        youdens_j([1], [0, 1])


def test_extra_has_components():
    r = youdens_j([1, 0, 1, 0], [1, 0, 0, 0])
    assert "sensitivity" in r.extra
    assert "specificity" in r.extra
