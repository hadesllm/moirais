eBAC — Estimated Blood Alcohol Concentration
=============================================

eBAC is a continuous outcome derived from self-reported alcohol consumption
data in CPADS. MOIRAIS computes two eBAC variants.

Widmark formula
---------------

The standard Widmark (1932) formula estimates BAC from consumed alcohol:

.. math::

   \text{BAC} = \frac{A}{r \cdot W} - \beta \cdot t

where

- :math:`A` = grams of alcohol consumed
- :math:`r` = Widmark distribution factor (0.68 for males, 0.55 for females)
- :math:`W` = body weight (kg)
- :math:`\beta` = elimination rate (≈ 0.15 g/dL/hr)
- :math:`t` = hours since drinking began

MOIRAIS variants
-------------

``ebac_tot``
   Total eBAC from the full CPADS drinking episode as reported.

``ebac_legal``
   Binary indicator: :math:`\mathbb{1}[\text{eBAC} \geq 0.08\text{ g/dL}]`
   (Canadian legal driving limit).

Both are available as canonical CPADS variables and participate in the
``CPADS_REQUIRED_VARIABLES`` contract.

Python API
----------

.. code-block:: python

   from moirais import calculate_ebac, is_over_legal_limit

   ebac = calculate_ebac(drinks=5, weight_kg=70, gender="male", hours=2.0)
   over = is_over_legal_limit(ebac)

eBAC-IPW module
---------------

The ``ebac-selection-adjustment-ipw`` module uses eBAC strata as a
selection-correction mechanism. See :doc:`causal` for the statistical
framework.

References
----------

- Widmark EMP (1932). *Die theoretischen Grundlagen und die praktische
  Verwendbarkeit der gerichtlich-medizinischen Alkoholbestimmung*.
  Urban & Schwarzenberg.
- Brick J (2006). Standardization of alcohol calculations in research.
  *Alcoholism: Clinical and Experimental Research*, 30(8):1276–1287.
  https://doi.org/10.1111/j.1530-0277.2006.00155.x
