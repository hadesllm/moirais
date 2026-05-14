"""
morie.entheo -- Psychedelic EEG-fMRI preprocessing and consciousness analysis.

Opt-in module; not auto-loaded by ``import morie``. Import explicitly
with ``import morie.entheo`` or ``from morie import entheo``.

Wraps the Carhart-Harris / Timmermann DMT-imaging data (20 subjects
EEG + parcellated fMRI; the 15 motion-survived subjects are 01-03 and
06-17, with 04, 05, 20, 21, 24 absent on disk per the manuscript)
into a coherent morie surface, and exposes two consciousness-theory
metrics:

  - **Beautiful Loop** (Bayne, Carter, Laukkonen, Slagter): predictive
    integration of phenomenal binding, scored as the cross-modal
    coupling between EEG-band power and fMRI gradient dispersion.
  - **Self-Aware Networks (SAN)** (Pirez): meta-cognitive recurrence,
    scored as the spectral slope of the autocorrelation of the
    EEG-fMRI joint state-vector.

Data location: the dataset is read from a local mirror at
``$MORIE_DMT_IMAGING_ROOT`` (default
``/Volumes/VSR/rootcoderfiles/DMT_Imaging``). When the mirror is
absent or ``scipy.io`` / ``pymatreader`` is unavailable, a
deterministic synthetic fixture is returned so downstream pipelines
keep running in CI.

Public API
----------
- ``load_dmt_imaging(subject_id=None)``
    BIDS-style record (EEG + fMRI + behavioural).
- ``preprocess_eeg(record, bandpass=(1, 40), notch=60, asr_threshold=20)``
    Butterworth bandpass + notch + ASR-style artifact rejection.
- ``preprocess_fmri(record, motion_threshold_mm=0.5)``
    Motion scrubbing + ICA-AROMA-style noise removal.
- ``beautiful_loop_metric(eeg, fmri)``
    Bayne-Laukkonen integrated phenomenal-binding score.
- ``san_score(eeg, fmri)``
    Self-Aware Network recurrence score.

R parity: every Python entrypoint above has an R sibling in the
``morie`` R package (``entheo_data.R``, ``entheo_preprocess.R``,
``entheo_analysis.R``).

References
----------
Timmermann, C. et al. (2023) Neural correlates of the DMT experience
  assessed with multivariate EEG. *Scientific Reports*, 13, 3147.
Bayne, T. & Carter, O. (2018) Dimensions of consciousness and the
  psychedelic state. *Neuroscience of Consciousness*, niy008.
Laukkonen, R. E. & Slagter, H. A. (2021) From many to (n)one:
  meditation and the plasticity of the predictive mind.
  *Neuroscience & Biobehavioral Reviews*, 128, 199-217.
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

from .. import __version__ as __version__
