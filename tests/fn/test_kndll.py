"""Tests for kendall_concordance."""
import numpy as np, pytest
from moirais.fn.kndll import kendall_concordance


class TestKendallConcordance:
    def test_perfect_agreement(self):
        ratings = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
        r = kendall_concordance(ratings)
        assert r.test_name == "Kendall's W"
        assert r.statistic == pytest.approx(1.0)

    def test_no_agreement(self):
        ratings = np.array([[1, 2, 3], [3, 2, 1]])
        r = kendall_concordance(ratings)
        assert r.statistic < 0.5

    def test_bad_shape(self):
        with pytest.raises(ValueError):
            kendall_concordance(np.array([1, 2, 3]))
