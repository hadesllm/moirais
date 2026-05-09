"""Test cross-entropy loss."""
import numpy as np
from moirais.fn.xenty import xenty


def test_xenty_binary():
    """Test binary cross-entropy."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.8, 0.2])
    loss = xenty(y_true, y_pred)
    assert loss > 0


def test_xenty_from_logits():
    """Test with logits."""
    y_true = np.array([0, 1])
    y_pred = np.array([0.0, 2.0])
    loss = xenty(y_true, y_pred, from_logits=True)
    assert loss > 0


def test_xenty_multiclass():
    """Test multi-class cross-entropy."""
    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    loss = xenty(y_true, y_pred)
    assert loss > 0 and loss < 10


def test_xenty_perfect_prediction():
    """Test perfect prediction."""
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[1.0 - 1e-8, 1e-8], [1e-8, 1.0 - 1e-8]])
    loss = xenty(y_true, y_pred)
    assert loss < 1e-6


def test_xenty_shape_mismatch():
    """Test shape mismatch."""
    try:
        xenty(np.array([0, 1]), np.array([0.5]))
        assert False
    except ValueError:
        pass
