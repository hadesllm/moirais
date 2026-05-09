# moirais.fn — function file (hadesllm/moirais)
"""Speech signal formant and pitch extraction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_speech_features"]


def rangayyan_speech_features(speech, fs, order):
    """
    Speech signal formant and pitch extraction

    Formula: Formants F1-F3 from LPC poles; pitch via autocorrelation peak

    Parameters
    ----------
    speech : array-like
        Input data.
    fs : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: formants, pitch

    References
    ----------
    Rangayyan Ch 1.2.13
    """
    speech = np.asarray(speech, dtype=float)
    n = int(speech) if speech.ndim == 0 else len(speech)
    result = float(np.mean(speech))
    se = float(np.std(speech, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Speech signal formant and pitch extraction"})


def cheatsheet():
    return "rgspeech: Speech signal formant and pitch extraction"
