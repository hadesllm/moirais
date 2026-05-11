What is statistics actually for?
================================

This page exists because most stats courses start with a formula and
expect you to figure out *why* later.  The why comes first here.

----

The one-sentence answer
-----------------------

**Statistics is the science of being honest about what you don't know.**

If you flip a coin three times and it comes up heads three times, two
people can give you very different answers about whether the coin is
fair:

- A storyteller says "all heads, must be rigged".
- A statistician says "yes, all heads — but I'd see this 1 in 8 times
  with a fair coin, so I am not yet sure".

The statistician's answer is *less* exciting and *more* useful.
Statistics is the discipline of keeping the second answer instead of
the first one when the data is too thin to support a confident claim.

----

The three things every analysis must do
---------------------------------------

Whether you're testing a drug or comparing two web designs or
estimating the population of fish in a lake, the work is the same
three steps:

1. **State the question precisely.**  "Does the drug work?" is too
   vague.  "Is the 30-day mortality rate of patients given drug A
   different from those given drug B, by more than 1 percentage
   point?" — that's a question.

2. **Pick an estimator.**  An estimator is a recipe that turns data
   into a single number that's supposed to answer the question.
   For the drug question, the estimator might be "the difference in
   mean mortality between the two groups, weighted by inverse
   probability of treatment to handle that the groups weren't
   randomly assigned".

3. **Quantify how wrong it might be.**  This is the standard error,
   the confidence interval, the p-value.  Every honest answer ends
   in *plus or minus something*.  An answer without that "plus or
   minus" is a guess wearing a hat.

MORIE has functions for each of those three steps.  The
``fn/`` tree (10,000+ small files, each one a single formula) is the
estimator vocabulary; the modules in ``morie.causal``,
``morie.effects``, ``morie.inference`` are the recipes that
combine them.

----

The four kinds of question this software is good at
----------------------------------------------------

1. **Did X cause Y?**  Causal inference.  IPW, propensity scores,
   double machine learning, instrumental variables.  See
   :doc:`../methods/causal` once you're past this track.

2. **Are these two groups actually different?**  Hypothesis testing.
   t-tests, chi-square, robust alternatives when the data is messy.
   :doc:`../methods/inference_engine`.

3. **What's the population value?**  Estimation.  Means, proportions,
   regression coefficients, with proper confidence intervals.

4. **Where is X higher than usual?**  Spatial / temporal pattern
   detection.  Moran's I, kriging, change-point detection.

You don't have to read all four.  Pick the one that matches the
question you walked in with.

----

The four kinds of question this software is **NOT** good at
-----------------------------------------------------------

To save you time:

- **Making pretty charts.**  Use matplotlib, ggplot2, or D3.  We
  produce numbers, not graphics.  The numbers feed your charts.
- **Cleaning a dataset.**  We assume the dataset is already in the
  shape you want.  Use pandas / dplyr for the wrangling step.
- **Predicting the future from a tweet.**  Forecasting needs
  specific time-series tooling.  We have some (``morie.fn``
  TimeSeries category), but Prophet / statsforecast / GluonTS are
  more focused if forecasting is your only goal.
- **Replacing a stats degree.**  If the question is hard, the answer
  is usually nuanced — a tool that gives you "the answer" with no
  caveats is lying to you.

----

What to read next
-----------------

- :doc:`first-analysis` — walk through one full analysis end-to-end.
  ~15 minutes.  Uses the iris dataset, no installation pain.
- :doc:`../install` — when you're ready to install MORIE for real.
- :doc:`../methods/index` — the methods reference, organised by
  question rather than by function name.
