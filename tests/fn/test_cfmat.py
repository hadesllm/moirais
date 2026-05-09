"""Tests for cfmat.py - Confusion matrix metrics."""
import numpy as np
import pytest
from moirais.fn.cfmat import cfmat_fn, cfmat


def test_cfmat_returns_descriptive_result():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    result = cfmat_fn(y_true, y_pred)
    assert result.name == "confusion_matrix"
    assert "accuracy" in result.extra
    assert "confusion_matrix" in result.extra


def test_cfmat_perfect():
    y = np.array([0, 0, 1, 1])
    result = cfmat_fn(y, y)
    assert result.value == 1.0
    assert result.extra["sensitivity"] == 1.0
    assert result.extra["specificity"] == 1.0


def test_cfmat_binary_metrics():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 0])
    result = cfmat_fn(y_true, y_pred)
    assert "sensitivity" in result.extra
    assert "specificity" in result.extra
    assert "ppv" in result.extra
    assert "f1" in result.extra


def test_cfmat_alias():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    result = cfmat(y_true, y_pred)
    assert result.name == "confusion_matrix"
