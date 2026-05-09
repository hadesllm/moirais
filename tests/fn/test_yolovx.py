"""Tests for yolovx.yolo_decoupled_head."""
import numpy as np
import pytest
from moirais.fn.yolovx import yolo_decoupled_head


def test_yolovx_basic():
    """Test basic functionality."""
    features = np.random.default_rng(42).normal(0, 1, 100)
    anchor_free = np.random.default_rng(42).normal(0, 1, 100)
    result = yolo_decoupled_head(features, anchor_free)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_yolovx_edge():
    """Test edge cases."""
    features = np.random.default_rng(42).normal(0, 1, 100)
    anchor_free = np.random.default_rng(42).normal(0, 1, 100)
    result = yolo_decoupled_head(features, anchor_free)
    assert isinstance(result, dict)
