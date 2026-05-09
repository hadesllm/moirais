"""Test 1D convolution."""
import numpy as np
from moirais.fn.conv1 import conv1


def test_conv1_basic():
    """Test basic 1D convolution."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    kernel = np.array([0.5, 0.5])
    y = conv1(x, kernel, padding='valid')
    assert y.shape == (1, 3)


def test_conv1_batch():
    """Test batch input."""
    x = np.random.randn(4, 10)
    kernel = np.array([1.0, -1.0])
    y = conv1(x, kernel, padding='valid')
    assert y.shape == (4, 9)


def test_conv1_stride():
    """Test stride."""
    x = np.arange(10.0)
    kernel = np.array([1.0])
    y = conv1(x[np.newaxis, :], kernel, padding='valid', stride=2)
    assert y.shape[1] == 5
