From Zero — Start Here If You've Never Done This Before
=========================================================

This track assumes **you know nothing**.  Not "you've done a stats course
once and forgotten everything" — actually nothing.  No Python, no R, no
regression, no t-test.  You wandered into here and want to know what
this software is for and whether it's worth your time.

That is a fair question.  Most documentation is written by people who
already know the answer; that's why so much of it is useless to you.
This page is for the person who doesn't yet know the question.

----

Two questions you can answer by the end of this track
-----------------------------------------------------

.. raw:: html

   <ol style="font-size: 1.1em; line-height: 1.7;">
     <li>Given some data and a yes/no question (<em>"is this drug actually
         working?"</em>), how do I get an answer with a real margin of error
         instead of a vibe?</li>
     <li>How do I trust that answer when the data has the kind of mess
         every real dataset has — missing values, weird outliers, people
         who joined the study halfway through, regional variation that
         isn't the thing you're studying?</li>
   </ol>

If those questions sound exciting, the rest of this site is for you.
If they sound like "I just want to make a chart", that's fine too —
matplotlib is excellent and you don't need this.

----

What is MORIE
---------------

MORIE is a *toolbox*.  It's not magic, it's not a stats degree in a
box, it doesn't decide what you should do with your data.  It's a
collection of around ten thousand small functions, each one a single
formula from a published paper, plus a terminal interface that lets
you stitch them together into an analysis.

Three things make it different from a stats package:

1. **Every formula has a citation.**  If you run ``morie.fn.icc1``
   and want to know where the math came from, the answer is
   *Shrout & Fleiss (1979)* — printed right there in the help.  No
   black boxes.

2. **It runs on your machine.**  No cloud.  No API key.  No telemetry.
   The data stays where you put it; the analysis stays where you ran
   it.  This matters when the data is real (medical records, legal
   files, anything sensitive).

3. **It's the same tool for both languages most scientists use.**
   Python AND R, sharing the same dataset cache and the same
   functions.  If your collaborator uses one and you use the other,
   nobody has to convert anything.

----

What this track will not do
---------------------------

- Teach you how to write Python.  The
  `Python tutorial <https://docs.python.org/3/tutorial/>`_ is better
  at that than we are.  You only need to recognise these characters
  exist: ``= ( ) , .``
- Teach you statistics from scratch.  We will explain *which* tool
  to reach for and *why*, and link out to a serious explanation when
  the math gets thick.  We won't re-derive the central limit theorem.

----

The track
---------

.. toctree::
   :maxdepth: 1

   what-is-this
   first-analysis

----

If you're stuck
---------------

Drop a question in the GitHub issues — assume nothing about the
reader's background, that is what we're here for.
