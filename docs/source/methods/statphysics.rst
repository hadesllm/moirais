Statistical Physics of Crime
=============================

The ``moirais.tps_statphysics`` module collects four
statistical-physics models of urban crime that are routinely fit on
Toronto Police Service open-data feeds. The framing follows the
D'Orsogna-Perc (2015) review.

Methods
-------

Short-Brantingham reaction-diffusion PDE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Short, D'Orsogna, and Brantingham (2008) derive a reaction-diffusion
PDE for the *attractiveness field* :math:`A(\mathbf{x}, t)` --- a
scalar that tracks how attractive a location is to offenders. The
discretised lattice form admits a closed-form steady-state and
reproduces the empirically-observed *hot-spot patterns* in burglary
and theft.

- ``moirais.tps_statphysics.sdb_reaction_diffusion`` --- fits the
  diffusion coefficient :math:`\eta`, the activity rate
  :math:`\theta`, and the attractiveness decay :math:`\omega` from
  observed incident-density grids.
- Reference: Short, D'Orsogna, Brantingham, Schoenberg, Tita (2008).
  *A statistical model of criminal behavior.* MMMAS 18(suppl):1249--1267.

Lévy-flight tail index
~~~~~~~~~~~~~~~~~~~~~~

Brockmann-Hufnagel-Geisel (2006) showed that human displacement
distributions follow a power-law tail :math:`P(\Delta r) \sim
\Delta r^{-(1 + \alpha)}` --- a Lévy-flight signature. The same
estimator on inter-event spatial displacements between consecutive
crime incidents diagnoses whether offenders' movement is Brownian
(:math:`\alpha \ge 2`) or Lévy (:math:`\alpha < 2`).

- ``moirais.tps_statphysics.levy_flight_alpha`` --- Hill estimator on
  inter-event displacements with bootstrap CI.
- Reference: Brockmann, Hufnagel, Geisel (2006). *The scaling laws of
  human travel.* Nature 439:462--465.

Bettencourt urban scaling
~~~~~~~~~~~~~~~~~~~~~~~~~

Bettencourt, Lobo, West (2007) showed that many city-level metrics
:math:`Y` scale as a power of population :math:`N`:

.. math::

    Y(N) \;=\; Y_0\, N^{\beta}.

For *socio-economic* metrics (including crime totals),
:math:`\beta > 1` (super-linear). For *infrastructure* metrics
:math:`\beta < 1` (sub-linear). The estimator is HC3-robust OLS on
:math:`\log Y` vs. :math:`\log N`.

- ``moirais.tps_statphysics.urban_scaling_beta`` --- HC3-OLS fit of
  :math:`\beta` with sandwich SE.
- Reference: Bettencourt, Lobo, Helbing, Kühnert, West (2007).
  *Growth, innovation, scaling, and the pace of life in cities.*
  PNAS 104(17):7301--7306.

Lotka-Volterra predator-prey
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

D'Orsogna-Perc (2015, §3.4) frame the crime-and-policing dynamic as a
Lotka-Volterra system: crime totals as prey, police-officer counts as
predators. The fit recovers the four classical parameters
(:math:`\alpha, \beta, \gamma, \delta`) plus their stability
classification (stable focus / spiral / saddle).

- ``moirais.tps_statphysics.lotka_volterra_police_crime`` --- nonlinear
  least-squares on annual paired police / crime time series.
- Reference: D'Orsogna, Perc (2015). *Statistical physics of crime: A
  review.* Physics of Life Reviews 12:1--21, §3.4.

Companion methods (under :doc:`spatial`)
----------------------------------------

The following are spatial-statistics tools that the above
statistical-physics methods often pair with:

- Moran's :math:`I` global / local indicators of spatial autocorrelation
- Ripley's :math:`K` and :math:`L` for point-pattern second-order
  intensity
- Getis-Ord :math:`G^{*}` hot-spot tests
- DBSCAN density-based clustering
- Kulldorff space-time scan for significant spatio-temporal clusters
