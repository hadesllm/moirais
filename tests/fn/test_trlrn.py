"""Test transfer_learn (trlrn)."""
import numpy as np
from morie.fn.trlrn import transfer_learn, trlrn
from morie.fn._containers import DescriptiveResult


class TestTrlrn:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X_s = rng.standard_normal((40, 5))
        y_s = (X_s[:, 0] > 0).astype(int)
        X_t = rng.standard_normal((20, 5))
        result = transfer_learn(X_s, y_s, X_t, n_components=3)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "transfer_learn"
        assert len(result.extra["predictions"]) == 20

    def test_alignment_matrix_shape(self):
        rng = np.random.default_rng(42)
        X_s = rng.standard_normal((40, 5))
        y_s = (X_s[:, 0] > 0).astype(int)
        X_t = rng.standard_normal((20, 5))
        result = transfer_learn(X_s, y_s, X_t, n_components=3)
        assert result.extra["alignment_matrix"].shape == (3, 3)

    def test_alias(self):
        assert trlrn is transfer_learn
