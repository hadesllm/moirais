"""Tests for morie.fn.prcv — precision-recall curve."""
import numpy as np
import pytest
from morie.fn.prcv import pr_curve, prcv


def test_basic():
    r = pr_curve([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert "precision" in r.extra
    assert "recall" in r.extra
    assert "thresholds" in r.extra


def test_recall_starts_zero():
    r = pr_curve([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert r.extra["recall"][0] == 0.0


def test_precision_starts_one():
    r = pr_curve([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert r.extra["precision"][0] == 1.0


def test_alias():
    assert prcv is pr_curve


def test_length_mismatch():
    with pytest.raises(ValueError):
        pr_curve([1, 0], [0.5])


def test_no_positives():
    with pytest.raises(ValueError):
        pr_curve([0, 0, 0], [0.5, 0.6, 0.7])
