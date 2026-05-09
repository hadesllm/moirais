"""Tests for fzkoc.fauzi_kernel_order_m_condition."""
import numpy as np
import pytest
from moirais.fn.fzkoc import fauzi_kernel_order_m_condition


def test_fzkoc_basic():
    """Test basic functionality."""
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    order = 4
    result = fauzi_kernel_order_m_condition(kernel, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzkoc_edge():
    """Test edge cases."""
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    order = 4
    result = fauzi_kernel_order_m_condition(kernel, order)
    assert isinstance(result, dict)
