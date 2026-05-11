"""Tests for lstmb.py - LSTM for biosignals."""
import numpy as np
import pytest
from morie.fn.lstmb import lstmb_fn, lstmb


def test_lstmb_returns_descriptive_result():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 10))
    y = np.array([0.0]*15 + [1.0]*15)
    result = lstmb_fn(X, y, n_epochs=2, hidden_size=8)
    assert result.name == "lstm_biosignal"
    assert "accuracy" in result.extra
    assert "hidden_size" in result.extra


def test_lstmb_accuracy_bounded():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 8))
    y = np.array([0.0]*10 + [1.0]*10)
    result = lstmb_fn(X, y, n_epochs=2, hidden_size=4)
    assert 0.0 <= result.value <= 1.0


def test_lstmb_hidden_size_stored():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 8))
    y = np.array([0.0]*10 + [1.0]*10)
    result = lstmb_fn(X, y, hidden_size=16)
    assert result.extra["hidden_size"] == 16


def test_lstmb_alias():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 8))
    y = np.array([0.0]*10 + [1.0]*10)
    result = lstmb(X, y, n_epochs=1, hidden_size=4)
    assert result.name == "lstm_biosignal"
