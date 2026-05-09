"""Tests for gh_c14_25.ghosal_ibp_poisson."""
import numpy as np
import pytest
from moirais.fn.gh_c14_25 import ghosal_ibp_poisson


def test_gh_c14_25_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_ibp_poisson(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c14_25_edge():
    """Test edge cases."""
    result = ghosal_ibp_poisson(np.array([42.0]))
    assert result['n'] == 1
