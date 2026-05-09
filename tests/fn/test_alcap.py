"""Tests for alcap.alammar_image_captioning_pipeline."""
import numpy as np
import pytest
from moirais.fn.alcap import alammar_image_captioning_pipeline


def test_alcap_basic():
    """Test basic functionality."""
    img = np.random.default_rng(42).normal(0, 1, 100)
    visual_encoder = np.random.default_rng(42).normal(0, 1, 100)
    projector = np.random.default_rng(42).normal(0, 1, 100)
    llm = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_image_captioning_pipeline(img, visual_encoder, projector, llm)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alcap_edge():
    """Test edge cases."""
    img = np.random.default_rng(42).normal(0, 1, 100)
    visual_encoder = np.random.default_rng(42).normal(0, 1, 100)
    projector = np.random.default_rng(42).normal(0, 1, 100)
    llm = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_image_captioning_pipeline(img, visual_encoder, projector, llm)
    assert isinstance(result, dict)
