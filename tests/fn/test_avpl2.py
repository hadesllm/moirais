"""Test 2D average pooling."""
import numpy as np
from morie.fn.avpl2 import avpl2


def test_avpl2_basic():
    """Test basic average pooling."""
    x = np.arange(16.0).reshape(1, 4, 4)
    y = avpl2(x, pool_size=2)
    assert y.shape == (1, 2, 2)


def test_avpl2_values():
    """Test correct average values."""
    x = np.array([[[1, 2], [3, 4]]])
    y = avpl2(x, pool_size=2)
    assert np.isclose(y[0, 0, 0], 2.5)
