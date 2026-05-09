"""Tests for goodman_kruskal_gamma."""
import numpy as np, pytest
from moirais.fn.gdman import goodman_kruskal_gamma


class TestGoodmanKruskalGamma:
    def test_concordant(self):
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        r = goodman_kruskal_gamma(x, y)
        assert r.measure == "goodman_kruskal_gamma"
        assert r.estimate == pytest.approx(1.0)

    def test_discordant(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        r = goodman_kruskal_gamma(x, y)
        assert r.estimate == pytest.approx(-1.0)

    def test_too_few(self):
        with pytest.raises(ValueError):
            goodman_kruskal_gamma([1], [2])
