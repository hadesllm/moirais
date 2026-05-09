"""Test top-k and top-p sampling."""
import numpy as np
from moirais.fn.topkp import topkp


def test_topkp_basic():
    """Test basic sampling."""
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = topkp(logits, top_k=3, top_p=0.9)
    assert 0 <= result["token_id"] < 5


def test_topkp_probabilities_sum():
    """Test probabilities sum to 1."""
    logits = np.random.randn(10)
    result = topkp(logits, top_k=5)
    assert np.isclose(np.sum(result["probabilities"]), 1.0)


def test_topkp_temperature():
    """Test temperature effect."""
    logits = np.array([1.0, 2.0, 3.0])
    result_hot = topkp(logits, temperature=10.0)
    result_cold = topkp(logits, temperature=0.1)
    assert np.sum(result_hot["probabilities"]) == np.sum(result_cold["probabilities"])
