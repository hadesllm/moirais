"""Tests for morie.fn.maccr — multi-class precision."""
import numpy as np
import pytest
from morie.fn.maccr import multiclass_precision, maccr


def test_perfect_macro():
    r = multiclass_precision([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2], average="macro")
    assert abs(r.estimate - 1.0) < 1e-10


def test_perfect_micro():
    r = multiclass_precision([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2], average="micro")
    assert abs(r.estimate - 1.0) < 1e-10


def test_perfect_weighted():
    r = multiclass_precision([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2], average="weighted")
    assert abs(r.estimate - 1.0) < 1e-10


def test_alias():
    assert maccr is multiclass_precision


def test_per_class_in_extra():
    r = multiclass_precision([0, 1, 0, 1], [0, 1, 0, 1], average="macro")
    assert "per_class" in r.extra
