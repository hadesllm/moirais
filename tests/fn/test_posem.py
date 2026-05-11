"""Test positional encoding."""
import numpy as np
from morie.fn.posem import posem


def test_posem_shape():
    """Test output shape."""
    pe = posem(100, 512)
    assert pe.shape == (100, 512)


def test_posem_bounded():
    """Test values are bounded."""
    pe = posem(100, 512)
    assert np.all(pe >= -1) and np.all(pe <= 1)


def test_posem_uniqueness():
    """Test different positions give different encodings."""
    pe = posem(10, 64)
    assert not np.allclose(pe[0], pe[1])
