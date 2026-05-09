"""moirais.signal -- Biomedical signal processing facade.

Re-exports all signal processing, fractal complexity, ECG/HRV,
PCG, and cepstral analysis functions from moirais.fn.*.

    from moirais.signal import buttlp, ecgdet, hfd, pcgmur
"""

from moirais.fn.buttbp import butter_bandpass as buttbp
from moirais.fn.buttbs import butter_bandstop as buttbs
from moirais.fn.butthp import butter_highpass as butthp
from moirais.fn.buttlp import butter_lowpass as buttlp
from moirais.fn.cepst import real_cepstrum as cepst
from moirais.fn.dfa import detrended_fluctuation as dfa
from moirais.fn.ecgdet import pan_tompkins as ecgdet
from moirais.fn.hcepst import complex_cepstrum as hcepst
from moirais.fn.hdecon import homomorphic_deconvolve as hdecon
from moirais.fn.hfd import higuchi_fd as hfd
from moirais.fn.hrvfd import hrv_freq_domain as hrvfd
from moirais.fn.hrvnl import hrv_nonlinear as hrvnl
from moirais.fn.hrvtd import hrv_time_domain as hrvtd
from moirais.fn.hurst import hurst
from moirais.fn.kfd import katz_fd as kfd
from moirais.fn.pburg import burg_psd as pburg
from moirais.fn.pcgenv import pcg_envelope as pcgenv
from moirais.fn.pcgflt import pcg_filter as pcgflt
from moirais.fn.pcgmur import pcg_murmur_score as pcgmur
from moirais.fn.pcgseg import pcg_segment as pcgseg
from moirais.fn.pfd import petrosian_fd as pfd
from moirais.fn.rrint import rr_intervals as rrint
from moirais.fn.sampen import sample_entropy as sampen
from moirais.fn.sgolay import savgol_smooth as sgolay
from moirais.fn.welch import welch_psd as welch

__all__ = [
    "buttlp",
    "butthp",
    "buttbp",
    "buttbs",
    "sgolay",
    "welch",
    "pburg",
    "hfd",
    "kfd",
    "pfd",
    "dfa",
    "sampen",
    "hurst",
    "ecgdet",
    "rrint",
    "hrvtd",
    "hrvfd",
    "hrvnl",
    "pcgflt",
    "pcgenv",
    "pcgseg",
    "pcgmur",
    "cepst",
    "hcepst",
    "hdecon",
]
