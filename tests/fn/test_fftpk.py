"""Tests for moirais.fn.fftpk."""
import numpy as np
from moirais.fn.fftpk import fft_peaks


def test_fftpk_smoke():
    rng = np.random.default_rng(42)
    result = fft_peaks(signal=np.sin(np.linspace(0, 4*np.pi, 100)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.fftpk import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
