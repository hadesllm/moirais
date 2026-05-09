"""Tests for ansari_bradley."""
import numpy as np, pytest
from moirais.fn.ansrb import ansari_bradley

class TestAnsari:
    def test_equal_scale(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 30)
        y = rng.normal(0, 1, 30)
        r = ansari_bradley(x, y)
        assert r.p_value > 0.05

    def test_diff_scale(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 50)
        y = rng.normal(0, 5, 50)
        r = ansari_bradley(x, y)
        assert r.test_name == "Ansari-Bradley"
