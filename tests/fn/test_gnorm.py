"""Test group normalization."""
import numpy as np
from moirais.fn.gnorm import gnorm


def test_gnorm_basic():
    """Test basic group norm."""
    x = np.random.randn(2, 32, 8, 8)
    result = gnorm(x, num_groups=8)
    assert result["output"].shape == x.shape


def test_gnorm_invalid_groups():
    """Test invalid groups."""
    x = np.random.randn(2, 32, 8, 8)
    try:
        gnorm(x, num_groups=7)
        assert False
    except ValueError:
        pass


def test_gnorm_shapes():
    """Test output shapes."""
    x = np.random.randn(4, 64, 16, 16)
    result = gnorm(x, num_groups=16)
    assert result["output"].shape == x.shape
    assert result["mean"].shape[:2] == (4, 16)
