"""Test global average pooling."""
import numpy as np
from morie.fn.gpool import gpool


def test_gpool_basic():
    """Test global average pooling."""
    x = np.arange(16.0).reshape(1, 4, 4)
    y = gpool(x)
    assert y.shape == (1,)


def test_gpool_batch():
    """Test with batch."""
    x = np.random.randn(4, 3, 8, 8)
    y = gpool(x)
    assert y.shape == (4, 3)


def test_gpool_values():
    """Test correct values."""
    x = np.array([[[1, 1], [1, 1]]])
    y = gpool(x)
    assert np.isclose(y[0], 1.0)
