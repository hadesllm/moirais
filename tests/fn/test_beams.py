"""Test beam search."""
import numpy as np
from moirais.fn.beams import beams


def test_beams_basic():
    """Test basic beam search."""
    def step_fn(tokens):
        return np.random.randn(100)

    result = beams(
        initial_token=0,
        step_fn=step_fn,
        max_length=10,
        beam_width=3,
        vocab_size=100,
    )
    assert len(result["sequences"]) == 3
    assert len(result["scores"]) == 3


def test_beams_sequences():
    """Test output sequences."""
    def step_fn(tokens):
        return np.random.randn(10)

    result = beams(
        initial_token=0,
        step_fn=step_fn,
        max_length=5,
        beam_width=2,
        vocab_size=10,
    )
    for seq in result["sequences"]:
        assert len(seq) == 5
