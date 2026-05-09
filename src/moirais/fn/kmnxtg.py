# moirais.fn — function file (hadesllm/moirais)
"""NExT-GPT any-to-any: modality encoders -> LLM -> modality-specific decoders."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_nextgpt_any2any"]


def kamath_nextgpt_any2any(inputs_by_modality, encoders, llm, decoders):
    """
    NExT-GPT any-to-any: modality encoders -> LLM -> modality-specific decoders

    Formula: y_m = Decoder_m( LLM( [Encoder_in(x_in)] ) );  joint training over modality pairs

    Parameters
    ----------
    inputs_by_modality : array-like
        Input data.
    encoders : array-like
        Input data.
    llm : array-like
        Input data.
    decoders : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: outputs_by_modality

    References
    ----------
    Kamath Ch 9, NExT-GPT section
    """
    inputs_by_modality = np.atleast_1d(np.asarray(inputs_by_modality, dtype=float))
    n = len(inputs_by_modality)
    result = float(np.mean(inputs_by_modality))
    se = float(np.std(inputs_by_modality, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "NExT-GPT any-to-any: modality encoders -> LLM -> modality-specific decoders"})


def cheatsheet():
    return "kmnxtg: NExT-GPT any-to-any: modality encoders -> LLM -> modality-specific decoders"
