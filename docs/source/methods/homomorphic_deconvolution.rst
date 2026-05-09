Homomorphic Deconvolution & Cepstral Analysis
==============================================

Cepstral analysis transforms convolution into addition via the log domain,
enabling separation of mixed signals (e.g., excitation from impulse response
in PCG recordings).

Real Cepstrum
-------------

The real cepstrum is defined as:

.. math::

   c[n] = \text{IFFT}\left(\log |X(k)|\right)

where :math:`X(k) = \text{FFT}(x[n])`.

.. code-block:: python

   from moirais.signal import cepst
   result = cepst(signal)   # CepstrumResult with .cepstrum array

Complex Cepstrum
----------------

The complex cepstrum preserves phase information:

.. math::

   \hat{x}[n] = \text{IFFT}\left(\log |X(k)| + j \cdot \text{unwrap}(\angle X(k))\right)

This is invertible: ``inverse_complex_cepstrum(complex_cepstrum(x)) = x``.

.. code-block:: python

   from moirais.signal import hcepst
   result = hcepst(signal)  # CepstrumResult

Liftering and Deconvolution
---------------------------

Given a signal :math:`y = h * e` (convolution of impulse response and
excitation), the complex cepstrum satisfies:

.. math::

   \hat{y}[n] = \hat{h}[n] + \hat{e}[n]

A low-time lifter (zeroing high-quefrency bins) isolates the slowly varying
minimum-phase component from the rapidly varying excitation.

.. code-block:: python

   from moirais.signal import hdecon

   result = hdecon(pcg_signal, cutoff_quefrency=64)
   min_phase = result.extra["min_phase"]
   residual = result.extra["residual"]

Application: PCG S1/S2 Decomposition
-------------------------------------

In phonocardiogram analysis, homomorphic deconvolution separates valve
closure transients (minimum-phase) from chest wall resonance. Combined
with Shannon-energy envelope analysis and Higuchi fractal dimension,
this enables murmur detection in cardiotoxicity studies.

.. code-block:: python

   from moirais.signal import pcgflt, hdecon, pcgmur

   filtered = pcgflt(pcg, fs=2000)
   decomposed = hdecon(filtered.filtered, cutoff_quefrency=128)
   score = pcgmur(pcg, fs=2000)

References
----------

- Oppenheim, A.V. & Schafer, R.W. (2009). *Discrete-Time Signal
  Processing*. Prentice Hall, Ch. 13.
- Rangayyan, R.M. (2015). *Biomedical Signal Analysis*. IEEE Press, p. 338.
