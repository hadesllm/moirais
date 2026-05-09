"""Tests for gh_c7_2.ghosal_kern_mix_kl."""
import numpy as np
import pytest
from moirais.fn.gh_c7_2 import ghosal_kern_mix_kl


def test_gh_c7_2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_kern_mix_kl(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c7_2_edge():
    """Test edge cases."""
    result = ghosal_kern_mix_kl(np.array([42.0]))
    assert result['n'] == 1
