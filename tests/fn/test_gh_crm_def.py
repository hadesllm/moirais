"""Tests for gh_crm_def.ghosal_completely_random_measure."""
import numpy as np
import pytest
from moirais.fn.gh_crm_def import ghosal_completely_random_measure


def test_gh_crm_def_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_completely_random_measure(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_crm_def_edge():
    """Test edge cases."""
    result = ghosal_completely_random_measure(np.array([42.0]))
    assert result['n'] == 1
