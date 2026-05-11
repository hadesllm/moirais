"""Tests for morie.fn.specf — specificity (TNR)."""
import numpy as np
import pytest
from morie.fn.specf import specificity, specf


def test_perfect():
    r = specificity([1, 0, 1, 0], [1, 0, 1, 0])
    assert abs(r.estimate - 1.0) < 1e-10


def test_half_specificity():
    r = specificity([0, 0, 1, 1], [0, 1, 1, 1])
    assert abs(r.estimate - 0.5) < 1e-10


def test_no_true_negatives():
    r = specificity([0, 0], [1, 1])
    assert abs(r.estimate - 0.0) < 1e-10


def test_alias():
    assert specf is specificity


def test_length_mismatch():
    with pytest.raises(ValueError):
        specificity([1], [0, 1])


def test_extra_keys():
    r = specificity([1, 0, 1, 0], [1, 1, 0, 0])
    assert r.extra["tn"] == 1
    assert r.extra["fp"] == 1
