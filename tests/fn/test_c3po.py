"""Tests for moirais.fn.c3po — Concordance index."""
import numpy as np

from moirais.fn.c3po import concordance_index, c3po


def test_perfect_concordance():
    """Perfect classifier should give C = 1.0."""
    y_true = [0, 0, 1, 1, 1]
    y_score = [0.1, 0.2, 0.7, 0.8, 0.9]
    result = concordance_index(y_true, y_score)
    assert result.estimate == 1.0
    assert result.extra["concordant"] > 0
    assert result.extra["discordant"] == 0


def test_random_concordance():
    """Random scores should give C near 0.5."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_score = rng.uniform(size=200)
    result = concordance_index(y_true, y_score)
    assert 0.3 < result.estimate < 0.7


def test_c3po_alias():
    assert c3po is concordance_index
