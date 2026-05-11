"""Test dropout."""
import numpy as np
from morie.fn.drupt import drupt


def test_drupt_training():
    """Test dropout during training."""
    x = np.ones((10, 10))
    y = drupt(x, rate=0.5, training=True, seed=42)
    assert y.shape == x.shape
    assert not np.allclose(y, x)


def test_drupt_inference():
    """Test no dropout during inference."""
    x = np.ones((10, 10))
    y = drupt(x, rate=0.5, training=False)
    assert np.allclose(y, x)


def test_drupt_zero_rate():
    """Test zero dropout rate."""
    x = np.ones((10, 10))
    y = drupt(x, rate=0.0, training=True)
    assert np.allclose(y, x)


def test_drupt_scaling():
    """Inverted-dropout scales surviving units by 1/(1-rate) so E[y]≈E[x]."""
    x = np.ones((100, 100))
    y = drupt(x, rate=0.5, training=True, seed=42)
    assert np.isclose(np.mean(y), np.mean(x), rtol=0.05)
