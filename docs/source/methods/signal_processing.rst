Signal Processing & Biomedical Analysis
========================================

MOIRAIS provides 25 biomedical signal processing functions via ``moirais.signal``
and individual ``moirais.fn.*`` modules. All functions are dataset-agnostic:
numpy arrays in, result objects out.

Digital Filters
---------------

Butterworth zero-phase filters (via ``scipy.signal``):

.. code-block:: python

   from moirais.signal import buttlp, butthp, buttbp, buttbs, sgolay

   result = buttlp(ecg_signal, fs=500, cutoff=40, order=4)
   filtered = result.filtered

- ``buttlp`` -- Lowpass
- ``butthp`` -- Highpass
- ``buttbp`` -- Bandpass
- ``buttbs`` -- Bandstop (notch, default 59-61 Hz for mains hum)
- ``sgolay`` -- Savitzky-Golay polynomial smoothing

Spectral Analysis
-----------------

.. code-block:: python

   from moirais.signal import welch, pburg

   psd = welch(signal, fs=256)        # Welch PSD
   ar_psd = pburg(signal, order=16)   # Burg AR PSD (parametric)

Fractal Complexity
------------------

Pure-numpy implementations for nonlinear time-series characterization:

.. code-block:: python

   from moirais.signal import hfd, kfd, pfd, dfa, sampen, hurst

   result = hfd(signal, kmax=10)    # Higuchi fractal dimension
   alpha = dfa(signal).value        # DFA scaling exponent

- ``hfd`` -- Higuchi fractal dimension (Higuchi, 1988)
- ``kfd`` -- Katz fractal dimension
- ``pfd`` -- Petrosian fractal dimension
- ``dfa`` -- Detrended fluctuation analysis
- ``sampen`` -- Sample entropy
- ``hurst`` -- Hurst exponent (R/S analysis)

Typical values: Brownian motion HFD ~1.5, white noise DFA alpha ~0.5.

ECG and Heart Rate Variability
------------------------------

.. code-block:: python

   from moirais.signal import ecgdet, rrint, hrvtd, hrvfd, hrvnl

   peaks = ecgdet(ecg, fs=360)              # Pan-Tompkins QRS detection
   rr = rrint(peaks.extra["qrs_indices"], fs=360)
   td = hrvtd(rr)                            # SDNN, RMSSD, pNN50
   fd = hrvfd(rr)                            # VLF/LF/HF power
   nl = hrvnl(rr)                            # Poincare SD1/SD2

Pan-Tompkins detector (Pan & Tompkins, 1985) with adaptive thresholding.
HRV metrics follow Task Force (1996) standards.

Phonocardiogram (PCG) Analysis
------------------------------

For cardiotoxicity studies in addiction/substance use research:

.. code-block:: python

   from moirais.signal import pcgflt, pcgenv, pcgseg, pcgmur

   filtered = pcgflt(pcg, fs=2000)           # 25-400 Hz bandpass
   envelope = pcgenv(pcg, fs=2000)           # Shannon energy envelope
   segments = pcgseg(envelope.filtered, fs=2000)  # S1/S2 segmentation
   score = pcgmur(pcg, fs=2000)              # Murmur detection score

The murmur detection score combines Higuchi fractal dimension, high-frequency
energy ratio, and spectral entropy. Scores are uncalibrated (0-1 range);
calibration requires labeled clinical data.

References
----------

- Higuchi, T. (1988). Approach to an irregular time series on the basis of
  the fractal theory. *Physica D*, 31(2), 277-283.
- Pan, J. & Tompkins, W.J. (1985). A real-time QRS detection algorithm.
  *IEEE Trans. Biomed. Eng.*, 32(3), 230-236.
- Task Force of ESC/NASPE (1996). Heart rate variability: Standards of
  measurement. *Circulation*, 93(5), 1043-1065.
- Rangayyan, R.M. (2015). *Biomedical Signal Analysis*. IEEE Press.
