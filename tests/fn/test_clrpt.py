"""Tests for morie.fn.clrpt — classification report."""
import numpy as np
import pytest
from morie.fn.clrpt import classification_report, clrpt


def test_perfect_binary():
    r = classification_report([1, 0, 1, 0], [1, 0, 1, 0])
    assert abs(r.value - 1.0) < 1e-10


def test_has_per_class_keys():
    r = classification_report([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2])
    assert "0" in r.extra
    assert "1" in r.extra
    assert "2" in r.extra


def test_macro_avg():
    r = classification_report([1, 0, 1, 0], [1, 0, 1, 0])
    assert abs(r.extra["macro_avg"]["f1"] - 1.0) < 1e-10


def test_weighted_avg():
    r = classification_report([1, 0, 1, 0], [1, 0, 1, 0])
    assert abs(r.extra["weighted_avg"]["f1"] - 1.0) < 1e-10


def test_alias():
    assert clrpt is classification_report


def test_per_class_support():
    r = classification_report([0, 0, 1], [0, 0, 1])
    assert r.extra["0"]["support"] == 2
    assert r.extra["1"]["support"] == 1
