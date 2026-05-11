"""Test ReLU6 activation."""
import numpy as np
from morie.fn.relu6 import relu6


def test_relu6_forward():
    """Test forward pass."""
    x = np.array([-2.0, 0.0, 3.0, 10.0])
    y = relu6(x)
    assert np.allclose(y, [0.0, 0.0, 3.0, 6.0])


def test_relu6_derivative():
    """Test gradient."""
    x = np.array([-2.0, 0.0, 3.0, 10.0])
    dy = relu6(x, derivative=True)
    assert np.allclose(dy, [0.0, 1.0, 1.0, 0.0])
