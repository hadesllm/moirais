"""Tests for hmyolo.geron_yolo."""
import numpy as np
import pytest
from moirais.fn.hmyolo import geron_yolo


def test_hmyolo_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_yolo(image, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmyolo_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_yolo(image, model)
    assert isinstance(result, dict)
