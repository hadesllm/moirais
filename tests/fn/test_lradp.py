"""Test learning rate scheduler."""
import numpy as np
from moirais.fn.lradp import lradp


def test_lradp_start():
    """Test LR at start."""
    lr = lradp(0, 100, learning_rate_max=0.1)
    assert np.isclose(lr, 0.1)


def test_lradp_end():
    """Test LR at end."""
    lr = lradp(100, 100, learning_rate_min=1e-6, learning_rate_max=0.1)
    assert np.isclose(lr, 1e-6)


def test_lradp_middle():
    """Test LR decreases."""
    lr0 = lradp(0, 100, learning_rate_max=0.1)
    lr50 = lradp(50, 100, learning_rate_max=0.1)
    lr100 = lradp(100, 100, learning_rate_max=0.1)
    assert lr0 > lr50 > lr100


def test_lradp_invalid_t():
    """Test invalid t."""
    try:
        lradp(150, 100)
        assert False
    except ValueError:
        pass


def test_lradp_invalid_lrs():
    """Test invalid learning rates."""
    try:
        lradp(0, 100, learning_rate_min=0.1, learning_rate_max=0.1)
        assert False
    except ValueError:
        pass
