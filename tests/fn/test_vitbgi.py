"""Tests for vitbgi.vit_b16_init."""
import numpy as np
import pytest
from morie.fn.vitbgi import vit_b16_init


def test_vitbgi_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = vit_b16_init(model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vitbgi_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = vit_b16_init(model)
    assert isinstance(result, dict)
