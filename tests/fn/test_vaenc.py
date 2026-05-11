"""Tests for vaenc.vae_elbo."""
import numpy as np
import pytest
from morie.fn.vaenc import vae_elbo


def test_vaenc_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = vae_elbo(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_vaenc_edge():
    """Test edge cases."""
    result = vae_elbo(np.array([42.0]))
    assert result['n'] == 1
