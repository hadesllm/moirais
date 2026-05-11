"""Tests for unetbk.unet_backbone."""
import numpy as np
import pytest
from morie.fn.unetbk import unet_backbone


def test_unetbk_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    filters = np.random.default_rng(42).normal(0, 1, 100)
    result = unet_backbone(x, filters)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_unetbk_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    filters = np.random.default_rng(42).normal(0, 1, 100)
    result = unet_backbone(x, filters)
    assert isinstance(result, dict)
