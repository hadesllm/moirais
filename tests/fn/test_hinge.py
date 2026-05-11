"""Test hinge loss."""
import numpy as np
from morie.fn.hinge import hinge


def test_hinge_basic():
    """Test basic hinge loss."""
    y_true = np.array([1, 1, -1, -1])
    y_pred = np.array([2.0, 0.5, -0.5, -2.0])
    loss = hinge(y_true, y_pred)
    assert loss >= 0


def test_hinge_perfect():
    """Test perfect classification."""
    y_true = np.array([1, -1])
    y_pred = np.array([2.0, -2.0])
    loss = hinge(y_true, y_pred, margin=1.0)
    assert loss == 0


def test_hinge_violated():
    """Test margin violation."""
    y_true = np.array([1, -1])
    y_pred = np.array([0.5, -0.5])
    loss = hinge(y_true, y_pred, margin=1.0)
    assert loss > 0


def test_hinge_invalid_labels():
    """Test invalid labels."""
    try:
        hinge(np.array([0, 2]), np.array([1.0, -1.0]))
        assert False
    except ValueError:
        pass


def test_hinge_invalid_margin():
    """Test invalid margin."""
    try:
        hinge(np.array([1, -1]), np.array([1.0, -1.0]), margin=-1)
        assert False
    except ValueError:
        pass
