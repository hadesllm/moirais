"""Tests for moirais.fn.ivw — IV Wald estimate."""

import pytest

from moirais.fn.ivw import iv_wald


class TestIVWald:
    def test_basic(self):
        res = iv_wald(0.5, 0.25, se_zy=0.1, se_zx=0.05)
        assert res.estimate == pytest.approx(2.0)

    def test_ci(self):
        res = iv_wald(0.5, 0.25, se_zy=0.1, se_zx=0.05)
        assert res.ci_lower < 2.0 < res.ci_upper

    def test_weak_instrument(self):
        with pytest.raises(ValueError):
            iv_wald(0.5, 0.0)
