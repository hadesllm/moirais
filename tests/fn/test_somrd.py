"""Tests for somers_d."""
import numpy as np, pytest
from moirais.fn.somrd import somers_d


class TestSomersD:
    def test_concordant(self):
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        r = somers_d(x, y)
        assert r.measure == "somers_d"
        assert r.estimate == pytest.approx(1.0)

    def test_discordant(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        r = somers_d(x, y)
        assert r.estimate == pytest.approx(-1.0)

    def test_too_few(self):
        with pytest.raises(ValueError):
            somers_d([1], [2])
