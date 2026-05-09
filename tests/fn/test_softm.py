"""Test softmax activation."""
import numpy as np
from moirais.fn.softm import softm


def test_softm_basic():
    """Test basic softmax."""
    x = np.array([1.0, 2.0, 3.0])
    y = softm(x)
    assert np.isclose(np.sum(y), 1.0)
    assert np.all(y >= 0) and np.all(y <= 1)


def test_softm_stability():
    """Test numerical stability with large values."""
    x = np.array([1000.0, 1001.0, 1002.0])
    y = softm(x)
    assert np.isclose(np.sum(y), 1.0)
    assert np.all(np.isfinite(y))


def test_softm_batch():
    """Test batch processing."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = softm(x, axis=-1)
    assert y.shape == x.shape
    assert np.allclose(np.sum(y, axis=-1), 1.0)
