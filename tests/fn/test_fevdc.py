"""Tests for moirais.fn.fevdc -- Forecast error variance decomposition."""
import numpy as np
import pytest
from moirais.fn.fevdc import fevd


class TestFEVD:
    def test_basic(self):
        A = np.array([[0.5, 0.1], [0.2, 0.3]])
        S = np.array([[1.0, 0.3], [0.3, 1.0]])
        res = fevd(A, S, periods=10)
        assert res.extra["decomposition"].shape == (11, 2, 2)
        row_sums = res.extra["decomposition"][-1].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            fevd(np.ones((2, 3)), np.eye(2))

    def test_cheatsheet(self):
        from moirais.fn.fevdc import cheatsheet
        assert isinstance(cheatsheet(), str)
