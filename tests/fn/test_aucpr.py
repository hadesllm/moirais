"""Tests for moirais.fn.aucpr — AUC-PR."""
import numpy as np
import pytest
from moirais.fn.aucpr import auc_pr, aucpr


def test_perfect_separation():
    r = auc_pr([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert r.estimate > 0.9


def test_value_between_zero_one():
    rng = np.random.default_rng(42)
    y = np.array([1] * 50 + [0] * 50)
    s = rng.uniform(size=100)
    r = auc_pr(y, s)
    assert 0.0 <= r.estimate <= 1.0


def test_alias():
    assert aucpr is auc_pr


def test_length_mismatch():
    with pytest.raises(ValueError):
        auc_pr([1, 0], [0.5])


def test_no_positives():
    with pytest.raises(ValueError):
        auc_pr([0, 0, 0], [0.5, 0.6, 0.7])


def test_extra_has_counts():
    r = auc_pr([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert r.extra["n_pos"] == 2
    assert r.extra["n_neg"] == 2
