"""Tests for semaeg.sam_image_encoder."""
import numpy as np
import pytest
from morie.fn.semaeg import sam_image_encoder


def test_semaeg_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    result = sam_image_encoder(image)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_semaeg_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    result = sam_image_encoder(image)
    assert isinstance(result, dict)
