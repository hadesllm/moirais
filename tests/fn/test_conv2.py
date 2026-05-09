"""Test 2D convolution."""
import numpy as np
from moirais.fn.conv2 import conv2


def test_conv2_basic():
    """Test basic 2D convolution."""
    x = np.random.randn(8, 8)
    kernel = np.ones((3, 3))
    y = conv2(x, kernel, padding='valid')
    assert y.shape == (1, 6, 6)


def test_conv2_batch():
    """Test batch input."""
    x = np.random.randn(2, 8, 8)
    kernel = np.ones((3, 3))
    y = conv2(x, kernel, padding='valid')
    assert y.shape == (2, 6, 6)


def test_conv2_same_padding():
    """Test same padding."""
    x = np.random.randn(8, 8)
    kernel = np.ones((3, 3))
    y = conv2(x, kernel, padding='same')
    assert y.shape == (1, 8, 8)
