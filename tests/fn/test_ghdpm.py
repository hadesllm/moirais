"""Tests for ghdpm.ghosal_dpmixture_density."""
import numpy as np
import pytest
from moirais.fn.ghdpm import ghosal_dpmixture_density


def test_ghdpm_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dpmixture_density(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ghdpm_edge():
    """Test edge cases."""
    result = ghosal_dpmixture_density(np.array([42.0]))
    assert result['n'] == 1
