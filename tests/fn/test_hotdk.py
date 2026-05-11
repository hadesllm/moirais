"""Tests for hot_deck_impute."""
import numpy as np, pytest
from morie.fn.hotdk import hot_deck_impute

class TestHotDeck:
    def test_fills_missing(self):
        data = np.array([[1, 2], [np.nan, 3], [4, np.nan]])
        r = hot_deck_impute(data, seed=0)
        assert r.extra["n_filled"] == 2

    def test_no_missing(self):
        data = np.ones((5, 3))
        r = hot_deck_impute(data, seed=0)
        assert r.value == 0
