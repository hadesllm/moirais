"""Tests for moirais.fn.mrecl — multi-class recall."""
import numpy as np
import pytest
from moirais.fn.mrecl import multiclass_recall, mrecl


def test_perfect_macro():
    r = multiclass_recall([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2], average="macro")
    assert abs(r.estimate - 1.0) < 1e-10


def test_perfect_micro():
    r = multiclass_recall([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2], average="micro")
    assert abs(r.estimate - 1.0) < 1e-10


def test_perfect_weighted():
    r = multiclass_recall([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2], average="weighted")
    assert abs(r.estimate - 1.0) < 1e-10


def test_alias():
    assert mrecl is multiclass_recall


def test_per_class_in_extra():
    r = multiclass_recall([0, 1, 0, 1], [0, 1, 0, 1], average="macro")
    assert "per_class" in r.extra
