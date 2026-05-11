"""Test layer normalization."""
import numpy as np
from morie.fn.lnorm import lnorm


def test_lnorm_basic():
    """Test basic layer norm."""
    x = np.random.randn(2, 4, 3)
    result = lnorm(x)
    assert result["output"].shape == x.shape
    assert np.isclose(np.mean(result["output"][0]), 0, atol=1e-6)


def test_lnorm_normalized():
    """Test output is normalized."""
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = lnorm(x)
    output = result["output"]
    assert np.isclose(np.std(output[0]), 1.0, atol=0.01)


def test_lnorm_with_scale():
    """Test with scale parameter."""
    x = np.random.randn(2, 4, 3)
    gamma = np.array([2.0, 2.0, 2.0])
    result = lnorm(x, gamma=gamma)
    assert result["output"].shape == x.shape


def test_lnorm_with_shift():
    """Test with shift parameter."""
    x = np.random.randn(2, 4, 3)
    beta = np.array([1.0, 1.0, 1.0])
    result = lnorm(x, beta=beta)
    assert result["output"].shape == x.shape
