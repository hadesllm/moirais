"""Tests for grdpml.geron_ddpm_simple_loss."""
import numpy as np
import pytest
from moirais.fn.grdpml import geron_ddpm_simple_loss


def test_grdpml_basic():
    """Test basic functionality."""
    eps = np.random.default_rng(42).normal(0, 1, 100)
    eps_pred = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ddpm_simple_loss(eps, eps_pred)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grdpml_edge():
    """Test edge cases."""
    eps = np.random.default_rng(42).normal(0, 1, 100)
    eps_pred = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ddpm_simple_loss(eps, eps_pred)
    assert isinstance(result, dict)
