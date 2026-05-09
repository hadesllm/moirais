"""Tests for ocrwit.ocr_wit_layout."""
import numpy as np
import pytest
from moirais.fn.ocrwit import ocr_wit_layout


def test_ocrwit_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    ocr_tokens = np.random.default_rng(42).normal(0, 1, 100)
    bboxes = np.random.default_rng(42).normal(0, 1, 100)
    result = ocr_wit_layout(image, ocr_tokens, bboxes)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ocrwit_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    ocr_tokens = np.random.default_rng(42).normal(0, 1, 100)
    bboxes = np.random.default_rng(42).normal(0, 1, 100)
    result = ocr_wit_layout(image, ocr_tokens, bboxes)
    assert isinstance(result, dict)
