Hawkes Self-Exciting Point Processes
=====================================

Part of :doc:`index` — MORIE's statistical-methods reference.

MORIE implements both the classical Markovian Hawkes process
(constant baseline, exponential excitation kernel) and the
non-stationary, non-Markovian generalisation of Kwan-Chen-Dunsmuir
(2024).

Modules
-------

- ``morie.tps_stochastic.hawkes_temporal_fit`` — classical Markovian
  Hawkes fit (one-parameter family, :math:`O(n)` recursive intensity).
- ``morie.tps_hawkes_advanced.fit_hawkes_general`` — MLE for the
  general (kernel, baseline) pair via adaptive Nelder-Mead.
- ``morie.tps_hawkes_advanced.compare_hawkes_kernels`` — fits all
  eight (kernel \times baseline) combinations and ranks by AIC and
  time-rescaling-residual Kolmogorov-Smirnov goodness-of-fit.
- ``morie.tps_hawkes_advanced.hawkes_markovian_vs_nonmarkovian`` —
  focused 2-way comparison: Markovian classical vs Gamma + sinusoidal.

Mathematical content
--------------------

For a simple point process :math:`N` on :math:`[0, T]` with conditional
intensity

.. math::

    \lambda(t) \;=\; \nu(t) \;+\; \int_{0}^{t-}\! g(t - s)\, dN_s,

the four supported excitation kernels :math:`g(u) = \eta\, \tilde
g(u; \psi)` (with branching ratio :math:`\eta \in (0, 1)`) are:

- **Exponential**: :math:`\tilde g_{\mathrm{exp}}(u; \beta) = \beta\,
  e^{-\beta u}` --- the classical Markovian case.
- **Gamma**: :math:`\tilde g_{\mathrm{gam}}(u; \alpha, \beta) =
  \beta^{\alpha}\, u^{\alpha - 1}\, e^{-\beta u} / \Gamma(\alpha)`.
- **Weibull**: :math:`\tilde g_{\mathrm{wb}}(u; \alpha, \lambda) =
  (\alpha / \lambda)\, (u / \lambda)^{\alpha - 1}\, e^{-(u /
  \lambda)^{\alpha}}`.
- **Lomax (power-law)**: :math:`\tilde g_{\mathrm{lmx}}(u; \alpha, c)
  = (\alpha - 1)\, c^{\alpha - 1}\, (u + c)^{-\alpha}` for
  :math:`\alpha > 1`.

The baseline takes the log-link form

.. math::

    \nu(t; \alpha) \;=\; \exp\!\left(\alpha_0 + \alpha_1\, t / T +
        \alpha_2 \sin(2 \pi t / 365.25) +
        \alpha_3 \cos(2 \pi t / 365.25)\right).

Inference is by maximum likelihood; standard errors via the numerical
Hessian. Goodness-of-fit by the Daley-Vere-Jones / Brown-Frank-Mitra
time-rescaling theorem.

Asymptotic theory
-----------------

Strong consistency and asymptotic normality of the MLE follow from
Kwan-Chen-Dunsmuir (2024). Under regularity conditions on
:math:`\nu(\cdot)` and :math:`\tilde g(\cdot)` (continuity, bounded
moments, :math:`\eta < 1`), the intensity process is asymptotically
ergodic and

.. math::

    \sqrt{n}\, (\hat\theta^n - \theta_0)
    \;\converginD\; \mathcal{N}\!\left(0,\; I(\theta_0)^{-1}\right).

Application
-----------

Applied to Toronto Police Service Assault data (post-2014,
:math:`n = 151{,}675` events), the four sinusoidal-baseline rows beat
every constant-baseline row. The best fit (Weibull kernel, sinusoidal
baseline) improves on the Markovian classical Hawkes by
:math:`\Delta\mathrm{AIC} = 141.1`. Branching-ratio estimates sit in
:math:`[0.83, 0.98]`.

Reference
---------

The full methodology and Toronto application are in:

- Ruhela, V. S. (2026). *Criminological Hawkes Process via MORIE:
  Markovian and Non-Markovian Self-Exciting Point Processes for
  Toronto Crime.* Zenodo. https://doi.org/10.5281/zenodo.20102198
