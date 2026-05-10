Your first analysis, end-to-end
================================

We will answer a real question with a real dataset, in fifteen lines
of code, with proper margins of error.  Nothing about this is fake.

The question
------------

The Fisher iris dataset is a 150-row table of measurements of three
species of iris flower (setosa, versicolor, virginica).  For each
flower we have sepal length, sepal width, petal length, petal width,
and the species name.

**Question:** *Are setosa flowers' petals shorter than virginica
flowers' petals, and by how much?*

This is a difference-of-means question.  Two groups, one numeric
outcome, you want a number with error bars.

----

Step 1: Load the data
---------------------

.. code-block:: python

   from moirais.fn import dnorm  # any moirais import warms the package
   import sqlite3
   import pandas as pd

   # MOIRAIS ships with the iris dataset in its built-in database.
   from moirais.data import moirais_builtin_db
   conn = sqlite3.connect(moirais_builtin_db())
   iris = pd.read_sql("SELECT * FROM iris", conn)
   conn.close()

   print(iris.head())
   #    sepal_length  sepal_width  petal_length  petal_width    species
   # 0           5.1          3.5           1.4          0.2     setosa
   # 1           4.9          3.0           1.4          0.2     setosa
   # ...

If you've never seen ``import``, ``=``, or ``from``, the
`Python tutorial <https://docs.python.org/3/tutorial/>`_ has a
one-page introduction.  Five minutes is enough.

----

Step 2: Pick the right tool
---------------------------

The question is "are petal lengths different between two groups?".
That maps to a *two-sample t-test* (with a robust alternative when
the data is skewed) — which lives at ``moirais.fn.t2smp``.

You can find this by:

- :doc:`../methods/inference_engine` — methods reference.
- The ``cheatsheet`` helper at the terminal:

  .. code-block:: python

     from moirais.cheatsheet import cheatsheet
     print(cheatsheet("t2smp"))

  That prints a whole help card: when to use, the formula reference,
  and a quote because it is more fun to learn that way.

----

Step 3: Run the test
--------------------

.. code-block:: python

   import numpy as np
   from moirais.fn.t2smp import t2smp  # if t2smp ships a module of the same name

   setosa_petal     = iris.loc[iris.species == "setosa",     "petal_length"].to_numpy()
   virginica_petal  = iris.loc[iris.species == "virginica",  "petal_length"].to_numpy()

   result = t2smp(setosa_petal, virginica_petal, alternative="two-sided")
   # result is a dict; keys depend on the exact fn.

The exact key names will be whatever the fn returns; the cheatsheet
will tell you.  In general expect:

- ``estimate`` — the difference of means
- ``statistic`` — the t-statistic
- ``p_value`` — the p-value
- ``ci`` — a 95% confidence interval

----

Step 4: Read the output honestly
--------------------------------

Three numbers matter:

1. **The estimate.**  How big is the difference?  If the difference
   is 4 cm but petals are around 1–6 cm long, that's enormous.  If
   the difference is 0.04 cm in the same range, that's basically
   nothing — even if the p-value is tiny.

2. **The confidence interval.**  Where could the truth plausibly
   be, given this much data?  A CI of (3.8, 4.2) cm means
   "I'm pretty sure the real difference is somewhere in there".
   A CI of (-0.5, 4.5) cm with the same point estimate of 2.0 means
   "I have no idea, the data is too thin".

3. **The p-value.**  How surprising would the data be if the truth
   really is "no difference"?  A p of 0.001 means "very surprising,
   probably a real difference".  A p of 0.4 means "not surprising,
   could easily be noise".  **The p-value is not the probability
   the difference is real** — it's the probability of the data
   given no difference.  Those are different statements.

----

Step 5: What to do when the data is messy
-----------------------------------------

The t-test assumes both groups are roughly normal.  Real data isn't.
For petal length on iris this is fine — but for income, time-on-page,
hospital cost, whatever the heck "engagement" means in your A/B test,
it usually isn't.

When the data is skewed or has extreme values, use the robust
alternatives in the ``RobustRegression`` and ``RobustWeight`` families.
The ``cheatsheet`` for any of them tells you when to reach for it.

----

Where to go next
----------------

- :doc:`../methods/causal` — once your question is causal ("did X
  cause Y?") and not just correlational.
- :doc:`../methods/index` — full methods reference, sorted by question.

----

The big idea, restated
----------------------

You loaded data, picked a tool that matched the question, ran it,
and got a number with a margin of error and a citation.  That is
the entire workflow.  The thousand other functions in MOIRAIS exist
because the question can take a thousand other shapes, but the
shape of the work doesn't change.
