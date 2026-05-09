"""Test Huber loss."""
import numpy as np
from moirais.fn.huber import huber


def test_huber_basic():
    """Test basic Huber loss."""
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    loss = huber(y_true, y_pred)
    assert loss >= 0


def test_huber_small_residuals():
    """Test quadratic region (small residuals)."""
    y_true = np.array([0, 1])
    y_pred = np.array([0.1, 0.9])
    loss = huber(y_true, y_pred, delta=1.0)
    assert loss > 0


def test_huber_large_residuals():
    """Test linear region (large residuals)."""
    y_true = np.array([0, 0])
    y_pred = np.array([10.0, -10.0])
    loss = huber(y_true, y_pred, delta=1.0)
    assert loss > 0


def test_huber_perfect():
    """Test perfect prediction."""
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    loss = huber(y_true, y_pred)
    assert np.isclose(loss, 0)


def test_huber_invalid_delta():
    """Test invalid delta."""
    try:
        huber(np.array([1]), np.array([1.1]), delta=-1)
        assert False
    except ValueError:
        pass
