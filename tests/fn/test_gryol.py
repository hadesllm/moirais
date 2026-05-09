"""Tests for gryol.geron_yolo_grid_loss."""
import numpy as np
import pytest
from moirais.fn.gryol import geron_yolo_grid_loss


def test_gryol_basic():
    """Test basic functionality."""
    predictions = np.random.default_rng(42).normal(0, 1, 100)
    targets = np.random.default_rng(42).normal(0, 1, 100)
    lam_coord = np.random.default_rng(42).normal(0, 1, 100)
    lam_noobj = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_yolo_grid_loss(predictions, targets, lam_coord, lam_noobj)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gryol_edge():
    """Test edge cases."""
    predictions = np.random.default_rng(42).normal(0, 1, 100)
    targets = np.random.default_rng(42).normal(0, 1, 100)
    lam_coord = np.random.default_rng(42).normal(0, 1, 100)
    lam_noobj = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_yolo_grid_loss(predictions, targets, lam_coord, lam_noobj)
    assert isinstance(result, dict)
