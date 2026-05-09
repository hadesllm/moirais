"""Tests for moirais.fn.mstar -- Markov-switching AR."""
import numpy as np
import pytest
from moirais.fn.mstar import ms_ar


class TestMSAR:
    def test_basic(self):
        rng = np.random.default_rng(42)
        y = np.concatenate([
            rng.normal(0, 1, 100),
            rng.normal(3, 1, 100),
        ])
        res = ms_ar(y, p=1, n_regimes=2, max_iter=20)
        assert len(res.extra["betas"]) == 2

    def test_short_raises(self):
        with pytest.raises(ValueError):
            ms_ar(np.ones(5), p=1)

    def test_cheatsheet(self):
        from moirais.fn.mstar import cheatsheet
        assert isinstance(cheatsheet(), str)
