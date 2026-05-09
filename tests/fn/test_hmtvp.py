"""Tests for hmtvp.geron_torchvision_pretrained."""
import numpy as np
import pytest
from moirais.fn.hmtvp import geron_torchvision_pretrained


def test_hmtvp_basic():
    """Test basic functionality."""
    model_name = 'default'
    n_classes = 3
    result = geron_torchvision_pretrained(model_name, n_classes)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmtvp_edge():
    """Test edge cases."""
    model_name = 'default'
    n_classes = 3
    result = geron_torchvision_pretrained(model_name, n_classes)
    assert isinstance(result, dict)
