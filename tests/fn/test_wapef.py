"""Test wapef."""
import numpy as np
import pytest
from morie.fn.wapef import wapef


def test_wapef_basic():
    """Weighted APE on predictions."""
    y_true = np.array([100, 200, 150, 300, 250])
    y_pred = np.array([95, 210, 145, 310, 240])

    r = wapef(y_true, y_pred)
    assert isinstance(r.value, float)
    assert isinstance(r.extra['wape'], float)
    assert isinstance(r.extra['mape'], float)


def test_wapef_perfect():
    """Perfect predictions."""
    y_true = np.array([10, 20, 30, 40, 50])
    y_pred = np.array([10, 20, 30, 40, 50])

    r = wapef(y_true, y_pred)
    assert r.value < 1e-6  # Should be near zero


def test_wapef_with_weights():
    """WAPE with custom weights."""
    y_true = np.array([100, 200, 150, 300])
    y_pred = np.array([90, 210, 140, 320])
    weights = np.array([1, 2, 1, 3])

    r = wapef(y_true, y_pred, weights=weights)
    assert isinstance(r.extra['wmape'], float)


def test_wapef_validation():
    """Input validation."""
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2])  # Wrong length

    with pytest.raises(ValueError):
        wapef(y_true, y_pred)
