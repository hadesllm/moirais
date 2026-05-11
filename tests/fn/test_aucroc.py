"""Tests for morie.fn.aucroc — AUC-ROC."""
import numpy as np
import pytest
from morie.fn.aucroc import auc_roc, aucroc


def test_perfect_separation():
    r = auc_roc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert abs(r.estimate - 1.0) < 1e-10


def test_random():
    rng = np.random.default_rng(42)
    y = np.array([1] * 500 + [0] * 500)
    s = rng.uniform(size=1000)
    r = auc_roc(y, s)
    assert abs(r.estimate - 0.5) < 0.1


def test_inverse():
    r = auc_roc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9])
    assert abs(r.estimate - 0.0) < 1e-10


def test_alias():
    assert aucroc is auc_roc


def test_length_mismatch():
    with pytest.raises(ValueError):
        auc_roc([1, 0], [0.5])


def test_all_same_class():
    with pytest.raises(ValueError):
        auc_roc([1, 1, 1], [0.5, 0.6, 0.7])
