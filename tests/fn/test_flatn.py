"""Test flatten layer."""
import numpy as np
from moirais.fn.flatn import flatn


def test_flatn_basic():
    """Test basic flattening."""
    x = np.random.randn(2, 3, 4)
    y = flatn(x)
    assert y.shape == (2, 12)


def test_flatn_2d():
    """Test 2D input."""
    x = np.random.randn(5, 10)
    y = flatn(x)
    assert y.shape == (5, 10)
