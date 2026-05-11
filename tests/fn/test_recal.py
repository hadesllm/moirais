"""Tests for morie.fn.recal — recall (sensitivity / TPR)."""
import numpy as np
import pytest
from morie.fn.recal import recall, recal


def test_perfect():
    r = recall([1, 0, 1, 0], [1, 0, 1, 0])
    assert abs(r.estimate - 1.0) < 1e-10


def test_half_recall():
    r = recall([1, 1, 0, 0], [1, 0, 0, 0])
    assert abs(r.estimate - 0.5) < 1e-10


def test_no_true_positives():
    r = recall([1, 1], [0, 0])
    assert abs(r.estimate - 0.0) < 1e-10


def test_alias():
    assert recal is recall


def test_length_mismatch():
    with pytest.raises(ValueError):
        recall([1], [0, 1])


def test_extra_keys():
    r = recall([1, 1, 0, 0], [1, 0, 1, 0])
    assert r.extra["tp"] == 1
    assert r.extra["fn"] == 1
