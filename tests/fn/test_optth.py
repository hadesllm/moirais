"""Tests for morie.fn.optth — optimal threshold selection."""
import numpy as np
import pytest
from morie.fn.optth import optimal_threshold, optth


def test_youden_perfect():
    r = optimal_threshold([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert r.extra["youdens_j"] > 0.9


def test_cost_method():
    r = optimal_threshold([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], method="cost")
    assert 0.0 <= r.estimate <= 1.0


def test_returns_threshold_in_range():
    r = optimal_threshold([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    scores = [0.9, 0.8, 0.2, 0.1]
    assert min(scores) <= r.estimate <= max(scores)


def test_alias():
    assert optth is optimal_threshold


def test_length_mismatch():
    with pytest.raises(ValueError):
        optimal_threshold([1, 0], [0.5])


def test_all_same_class():
    with pytest.raises(ValueError):
        optimal_threshold([1, 1, 1], [0.5, 0.6, 0.7])


def test_extra_has_sensitivity():
    r = optimal_threshold([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert "sensitivity" in r.extra
    assert "specificity" in r.extra
