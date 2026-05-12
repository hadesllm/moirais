"""
morie.entheo -- Psychedelic EEG-fMRI preprocessing and consciousness analysis.

v0.4.0-alpha scaffold. Opt-in module; not auto-loaded by `import morie`.

Wraps Dr. Robin Carhart-Harris / Christopher Timmermann DMT-imaging
data (20 subjects EEG + parcellated fMRI; 15 subjects survived motion
correction) into a coherent morie surface, and provides toy
implementations of two consciousness-theory metrics:

  - Beautiful Loop  (Bayne, Carter, Laukkonen, Slagter): predictive
    integration of phenomenal binding, here scored as the cross-modal
    coupling between EEG-band power and fMRI gradient dispersion.
  - Self-Aware Networks (SAN, Carlos Pirez): meta-cognitive recurrence,
    scored as the spectral slope of the autocorrelation of the
    EEG-fMRI joint state-vector.

Toy in v0.4.0-alpha -- actual psychometric calibration in v0.4.0-rc1.

Public API
----------
- ``load_dmt_imaging(subject_id=None)``
    BIDS-style record (EEG + fMRI + behavioural).
- ``preprocess_eeg(record, bandpass=(1, 40), notch=60, asr_threshold=20)``
    Butterworth bandpass + notch + ASR-style artifact rejection.
- ``preprocess_fmri(record, motion_threshold_mm=0.5)``
    Motion scrubbing + ICA-AROMA-style noise removal (toy).
- ``beautiful_loop_metric(eeg, fmri)``
    Bayne-Laukkonen integrated phenomenal-binding score.
- ``san_score(eeg, fmri)``
    Self-Aware Network recurrence score.

References
----------
Timmermann, C. et al. (2023) Neural correlates of the DMT experience
  assessed with multivariate EEG. *Scientific Reports*, 13, 3147.
Bayne, T. & Carter, O. (2018) Dimensions of consciousness and the
  psychedelic state. *Neuroscience of Consciousness*, niy008.
Laukkonen, R. E. & Slagter, H. A. (2021) From many to (n)one:
  meditation and the plasticity of the predictive mind.
  *Neuroscience & Biobehavioral Reviews*, 128, 199-217.

Co-Authored-By: Yoda <noreply@anthropic.com>
"""

from __future__ import annotations

from .data import load_dmt_imaging
from .preprocess import preprocess_eeg, preprocess_fmri
from .analysis import beautiful_loop_metric, san_score

__all__ = [
    "load_dmt_imaging",
    "preprocess_eeg",
    "preprocess_fmri",
    "beautiful_loop_metric",
    "san_score",
]

__version__ = "0.4.0-alpha"
