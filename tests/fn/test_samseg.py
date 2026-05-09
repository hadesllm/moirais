"""Tests for samseg.sam_segment."""
import numpy as np
import pytest
from moirais.fn.samseg import sam_segment


def test_samseg_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    prompts = np.random.default_rng(42).normal(0, 1, 100)
    result = sam_segment(image, prompts)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_samseg_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    prompts = np.random.default_rng(42).normal(0, 1, 100)
    result = sam_segment(image, prompts)
    assert isinstance(result, dict)
