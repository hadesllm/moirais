"""Tests for contRC.content_based."""
import numpy as np
import pytest
from morie.fn.contRC import content_based


def test_contRC_basic():
    """Test basic functionality."""
    item_feat = np.random.default_rng(42).normal(0, 1, 100)
    user_profile = np.random.default_rng(42).normal(0, 1, 100)
    result = content_based(item_feat, user_profile)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_contRC_edge():
    """Test edge cases."""
    item_feat = np.random.default_rng(42).normal(0, 1, 100)
    user_profile = np.random.default_rng(42).normal(0, 1, 100)
    result = content_based(item_feat, user_profile)
    assert isinstance(result, dict)
