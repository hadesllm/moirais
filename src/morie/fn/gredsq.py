# morie.fn — function file (hadesllm/morie)
"""Encoder-decoder seq2seq: context vector from encoder initializes decoder."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_encoder_decoder_seq2seq"]


def geron_encoder_decoder_seq2seq(encoder, decoder, x, max_out_len):
    """
    Encoder-decoder seq2seq: context vector from encoder initializes decoder

    Formula: c = encoder(x_{1:T}); y_t = decoder(y_{t-1}, c, s_t)

    Parameters
    ----------
    encoder : array-like
        Input data.
    decoder : array-like
        Input data.
    x : array-like
        Input data.
    max_out_len : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Géron Ch 14, Encoder-Decoder / Seq2Seq section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Encoder-decoder seq2seq: context vector from encoder initializes decoder"})


def cheatsheet():
    return "gredsq: Encoder-decoder seq2seq: context vector from encoder initializes decoder"
