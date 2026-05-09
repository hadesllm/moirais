"""Tests for gh_contr_rate2.ghosal_contraction_rate_iid."""
import numpy as np
import pytest
from moirais.fn.gh_contr_rate2 import ghosal_contraction_rate_iid


def test_gh_contr_rate2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_contraction_rate_iid(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_contr_rate2_edge():
    """Test edge cases."""
    result = ghosal_contraction_rate_iid(np.array([42.0]))
    assert result['n'] == 1
