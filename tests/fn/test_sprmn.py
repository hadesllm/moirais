"""Tests for spearman_corr."""
import numpy as np, pytest
from morie.fn.sprmn import spearman_corr


class TestSpearmanCorr:
    def test_perfect(self):
        r = spearman_corr([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert r.measure == "spearman"
        assert r.estimate == pytest.approx(1.0)

    def test_inverse(self):
        r = spearman_corr([1, 2, 3, 4, 5], [10, 8, 6, 4, 2])
        assert r.estimate == pytest.approx(-1.0)

    def test_too_few(self):
        with pytest.raises(ValueError):
            spearman_corr([1, 2], [3, 4])
