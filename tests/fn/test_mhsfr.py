"""Tests for moirais.fn.mhsfr -- SF-12 mental component."""

import pytest
from moirais.fn.mhsfr import sf12_mental


class TestSF12Mental:
    def test_basic(self):
        items = [3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2]
        res = sf12_mental(items)
        assert res.measure == "SF-12_MCS"
        assert 0 <= res.estimate <= 100

    def test_wrong_len(self):
        with pytest.raises(ValueError):
            sf12_mental([1, 2, 3])
