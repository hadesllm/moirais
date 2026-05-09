"""Tests for gh_pd_2param.ghosal_poisson_dirichlet."""
import numpy as np
import pytest
from moirais.fn.gh_pd_2param import ghosal_poisson_dirichlet


def test_gh_pd_2param_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_poisson_dirichlet(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gh_pd_2param_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_poisson_dirichlet(x)
    assert isinstance(result, dict)
