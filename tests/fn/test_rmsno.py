"""Test RMS normalization."""
import numpy as np
from morie.fn.rmsno import rmsno


def test_rmsno_basic():
    """Test basic RMS norm."""
    x = np.random.randn(2, 4, 3)
    result = rmsno(x)
    assert result["output"].shape == x.shape


def test_rmsno_scale():
    """Test with scale parameter."""
    x = np.random.randn(2, 4, 3)
    gamma = np.array([2.0, 2.0, 2.0])
    result = rmsno(x, gamma=gamma)
    assert result["output"].shape == x.shape


def test_rmsno_stability():
    """Test numerical stability."""
    x = np.zeros((2, 3))
    result = rmsno(x, epsilon=1e-6)
    assert np.all(np.isfinite(result["output"]))
