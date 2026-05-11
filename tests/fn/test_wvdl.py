"""Test wavelet_dict_learn (wvdl)."""
import numpy as np
from morie.fn.wvdl import wavelet_dict_learn, wvdl
from morie.fn._containers import DescriptiveResult


class TestWvdl:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 20))
        result = wavelet_dict_learn(X, n_atoms=8, n_iter=3, sparsity=2)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "wavelet_dict_learn"
        assert result.value >= 0

    def test_dictionary_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((8, 15))
        r = wavelet_dict_learn(X, n_atoms=12, n_iter=2, sparsity=2)
        assert r.extra["dictionary"].shape == (8, 12)
        assert r.extra["coefficients"].shape == (12, 15)

    def test_wavelet_param(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((6, 10))
        r = wavelet_dict_learn(X, n_atoms=5, wavelet="haar", n_iter=2)
        assert r.extra["wavelet"] == "haar"

    def test_alias(self):
        assert wvdl is wavelet_dict_learn
