"""Test greedy decoding."""
import numpy as np
from moirais.fn.greed import greed


def test_greed_basic():
    """Test basic greedy decoding."""
    def step_fn(tokens):
        return np.random.randn(100)

    result = greed(
        initial_token=0,
        step_fn=step_fn,
        max_length=10,
        temperature=1.0,
    )
    assert len(result["sequence"]) == 10
    assert len(result["probabilities"]) == 9


def test_greed_temperature():
    """Test temperature parameter."""
    def step_fn(tokens):
        return np.array([1.0] * 10)

    result = greed(
        initial_token=0,
        step_fn=step_fn,
        max_length=5,
        temperature=1.0,
    )
    assert len(result["sequence"]) == 5
