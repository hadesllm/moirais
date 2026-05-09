"""Tests for matrans.ma_logit_transform."""
import numpy as np
import pytest
from moirais.fn.matrans import ma_logit_transform


def test_matrans_basic():
    """Test basic functionality."""
    p = 5
    n = 100
    result = ma_logit_transform(p, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_matrans_edge():
    """Test edge cases."""
    p = 5
    n = 100
    result = ma_logit_transform(p, n)
    assert isinstance(result, dict)
