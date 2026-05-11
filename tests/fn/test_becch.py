"""Tests for morie.fn.becch — BEC capacity."""

import pytest

from morie.fn.becch import becch


class TestBecch:
    def test_no_erasure(self):
        assert becch(0.0)["capacity"] == pytest.approx(1.0)

    def test_full_erasure(self):
        assert becch(1.0)["capacity"] == pytest.approx(0.0)

    def test_half_erasure(self):
        assert becch(0.5)["capacity"] == pytest.approx(0.5)

    def test_known_value(self):
        assert becch(0.3)["capacity"] == pytest.approx(0.7)

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError):
            becch(-0.1)
        with pytest.raises(ValueError):
            becch(1.1)

    def test_output_keys(self):
        result = becch(0.2)
        assert "capacity" in result
        assert "erasure_prob" in result
        assert result["erasure_prob"] == 0.2
