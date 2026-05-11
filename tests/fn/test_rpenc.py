"""Test rotary positional encoding."""
import numpy as np
from morie.fn.rpenc import rpenc


def test_rpenc_shape():
    """Test output shape."""
    rope = rpenc(100, 256)
    assert rope.shape == (100, 256)


def test_rpenc_bounded():
    """Test values are bounded."""
    rope = rpenc(100, 256)
    assert np.all(rope >= -1) and np.all(rope <= 1)


def test_rpenc_even_dim():
    """Test even dimension requirement."""
    try:
        rpenc(100, 63)
        assert False
    except ValueError:
        pass
