"""Tests for moirais.fn.balan — balanced accuracy."""
import numpy as np
import pytest
from moirais.fn.balan import balanced_accuracy, balan


def test_perfect():
    r = balanced_accuracy([1, 0, 1, 0], [1, 0, 1, 0])
    assert abs(r.estimate - 1.0) < 1e-10


def test_random():
    r = balanced_accuracy([1, 1, 0, 0], [1, 0, 1, 0])
    assert abs(r.estimate - 0.5) < 1e-10


def test_imbalanced():
    r = balanced_accuracy([1, 0, 0, 0, 0], [1, 0, 0, 0, 0])
    assert abs(r.estimate - 1.0) < 1e-10


def test_alias():
    assert balan is balanced_accuracy


def test_length_mismatch():
    with pytest.raises(ValueError):
        balanced_accuracy([1], [0, 1])


def test_extra_has_components():
    r = balanced_accuracy([1, 0, 1, 0], [1, 0, 1, 0])
    assert "sensitivity" in r.extra
    assert "specificity" in r.extra
