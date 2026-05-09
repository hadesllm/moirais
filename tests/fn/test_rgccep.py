"""Tests for rgccep.rangayyan_complex_cepstrum."""
import numpy as np
import pytest
from moirais.fn.rgccep import rangayyan_complex_cepstrum


def test_rgccep_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_complex_cepstrum(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgccep_edge():
    """Test edge cases."""
    result = rangayyan_complex_cepstrum(np.array([42.0]))
    assert result['n'] == 1
