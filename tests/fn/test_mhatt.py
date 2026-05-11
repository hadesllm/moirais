"""Test multi-head attention."""
import numpy as np
from morie.fn.mhatt import mhatt


def test_mhatt_basic():
    """Test basic attention."""
    q = np.random.randn(5, 512)
    k = np.random.randn(5, 512)
    v = np.random.randn(5, 512)
    result = mhatt(q, k, v, num_heads=8, d_model=512)
    assert result["output"].shape == q.shape


def test_mhatt_weights_sum():
    """Test attention weights sum to 1."""
    q = np.random.randn(4, 64)
    k = np.random.randn(4, 64)
    v = np.random.randn(4, 64)
    result = mhatt(q, k, v, num_heads=4, d_model=64)
    sums = np.sum(result["attention_weights"], axis=-1)
    assert np.allclose(sums, 1.0)


def test_mhatt_invalid_heads():
    """Test invalid num_heads."""
    q = np.random.randn(5, 64)
    k = np.random.randn(5, 64)
    v = np.random.randn(5, 64)
    try:
        mhatt(q, k, v, num_heads=5, d_model=64)
        assert False
    except ValueError:
        pass
