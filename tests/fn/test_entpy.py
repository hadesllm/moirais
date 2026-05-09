"""Tests for moirais.fn.entpy — Shannon entropy."""
import numpy as np
import pytest
from moirais.fn.entpy import entropy


class TestEntropy:
    def test_uniform(self):
        res = entropy(np.array([0.25, 0.25, 0.25, 0.25]))
        assert res.estimate == pytest.approx(2.0, abs=1e-10)

    def test_certain(self):
        res = entropy(np.array([1.0, 0.0, 0.0]))
        assert res.estimate == pytest.approx(0.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            entropy(np.array([-1, 2]))
