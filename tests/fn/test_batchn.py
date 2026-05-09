"""Tests for batchn (batch normalization)."""
import numpy as np
from moirais.fn.batchn import batch_norm


def test_batch_norm_basic():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r = batch_norm(x)
    output = np.array(r.extra["output"])
    assert abs(np.mean(output)) < 1e-10
    assert abs(np.std(output) - 1.0) < 0.1


def test_cheatsheet():
    from moirais.fn.batchn import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
