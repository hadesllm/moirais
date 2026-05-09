"""Tests for moirais.fn.roccv — ROC curve."""
import numpy as np
import pytest
from moirais.fn.roccv import roc_curve, roccv


def test_basic():
    r = roc_curve([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert "fpr" in r.extra
    assert "tpr" in r.extra
    assert "thresholds" in r.extra


def test_fpr_starts_zero():
    r = roc_curve([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert r.extra["fpr"][0] == 0.0


def test_tpr_starts_zero():
    r = roc_curve([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert r.extra["tpr"][0] == 0.0


def test_alias():
    assert roccv is roc_curve


def test_length_mismatch():
    with pytest.raises(ValueError):
        roc_curve([1, 0], [0.5])


def test_all_same_class():
    with pytest.raises(ValueError):
        roc_curve([1, 1, 1], [0.5, 0.6, 0.7])
