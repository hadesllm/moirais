"""Test 2D max pooling."""
import numpy as np
from morie.fn.mxpl2 import mxpl2


def test_mxpl2_basic():
    """Test basic max pooling."""
    x = np.arange(16.0).reshape(1, 4, 4)
    result = mxpl2(x, pool_size=2)
    assert result["output"].shape == (1, 2, 2)


def test_mxpl2_values():
    """Test correct max values."""
    x = np.array([[[1, 2], [3, 4]]])
    result = mxpl2(x, pool_size=2)
    assert result["output"][0, 0, 0] == 4


def test_mxpl2_mask():
    """Test mask generation."""
    x = np.arange(16.0).reshape(1, 4, 4)
    result = mxpl2(x, pool_size=2)
    assert result["mask"].shape == x.shape
