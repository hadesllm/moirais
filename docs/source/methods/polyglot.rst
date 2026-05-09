Polyglot REPL
=============

MOIRAIS includes a polyglot REPL that bridges multiple programming languages
within a single session. Variables, DataFrames, and scalars flow
bidirectionally between Python, R, and Shell, with additional support for
Julia, SQL, Node.js, Go, Rust, C, C++, OCaml, Lua, TypeScript, LaTeX, and
PostgreSQL — **15 languages** in one REPL.

Access the REPL via the TUI (``e`` key) or the CLI:

.. code-block:: bash

   moirais repl
   moirais repl --headless    # non-interactive, reads from stdin

Language Detection
------------------

The REPL auto-detects the language of each input line. Explicit prefixes
override auto-detection:

.. list-table::
   :header-rows: 1
   :widths: 15 15 40

   * - Prefix
     - Language
     - Display Label
   * - (none)
     - Python (default)
     - ``[P]``
   * - ``R>``
     - R
     - ``[R]``
   * - ``J>``
     - Julia
     - ``[J]``
   * - ``Q>``
     - SQL
     - ``[Q]``
   * - ``N>``
     - Node.js
     - ``[N]``
   * - ``Go>``
     - Go
     - ``[Go]``
   * - ``Rs>``
     - Rust
     - ``[Rs]``
   * - ``C>``
     - C
     - ``[C]``
   * - ``C+>``
     - C++
     - ``[C+]``
   * - ``ML>``
     - OCaml
     - ``[ML]``
   * - ``Lu>``
     - Lua
     - ``[Lu]``
   * - ``TS>``
     - TypeScript
     - ``[TS]``
   * - ``TX>``
     - LaTeX
     - ``[TX]``
   * - ``PG>``
     - PostgreSQL
     - ``[PG]``
   * - ``$`` or ``!``
     - Shell
     - ``[Z]`` (zsh), ``[B]`` (bash), ``[S]`` (sh)

Auto-detection heuristics:

- Lines starting with ``library(``, ``<-``, ``c(``, or ending with ``%>%``
  are classified as R
- Lines containing ``|``, ``&&``, ``cd``, ``ls``, ``grep`` are classified
  as Shell
- SQL keywords (``SELECT``, ``FROM``, ``WHERE``) trigger SQL mode
- ``package main``, ``fmt.Print`` → Go; ``println!``, ``let mut`` → Rust
- ``#include <`` → C; ``std::``, ``cout <<`` → C++
- ``Printf.printf``, ``List.map``, ``;;`` → OCaml
- ``local ``, ``pairs(``, ``ipairs(`` → Lua
- ``interface ``, ``: string``, ``: number`` → TypeScript
- ``\documentclass``, ``\begin{`` → LaTeX
- ``\dt``, ``\d+``, ``\conninfo`` → PostgreSQL
- Everything else defaults to Python

Compiled Languages
------------------

Go, Rust, C, C++, and OCaml code is compiled and executed in a temporary
directory. Each submission is a self-contained program — include ``main``
functions and necessary imports.

.. code-block:: text

   Go> package main
   Go> import "fmt"
   Go> func main() { fmt.Println("hello from Go") }

   Rs> fn main() { println!("hello from Rust"); }

   C> #include <stdio.h>
   C> int main() { printf("hello from C\n"); return 0; }

   ML> Printf.printf "hello from OCaml\n" ;;

Variable Bridging
-----------------

The ``/polyglot`` command enables full bidirectional variable bridging.
Scalars, vectors, strings, booleans, and DataFrames transfer across
language boundaries automatically.

**Python to R:**

.. code-block:: text

   [P] x = 42
   [P] name = "treatment"
   R> print(x)       # 42
   R> print(name)    # "treatment"

**R to Python:**

.. code-block:: text

   R> y <- c(1, 2, 3)
   [P] print(y)      # [1, 2, 3]

**DataFrame bridging:**

.. code-block:: text

   [P] import pandas as pd
   [P] df = pd.DataFrame({"a": [1,2,3], "b": [4,5,6]})
   R> summary(df)     # R sees the DataFrame
   R> result <- lm(b ~ a, data=df)

Supported types for bridging:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25

   * - Type
     - Python
     - R
   * - Scalar (int/float)
     - ``int``, ``float``
     - ``numeric``
   * - String
     - ``str``
     - ``character``
   * - Boolean
     - ``bool``
     - ``logical``
   * - Vector/List
     - ``list``, ``np.array``
     - ``c()``, ``vector``
   * - DataFrame
     - ``pd.DataFrame``
     - ``data.frame``

Modes
-----

The REPL supports three operational modes:

- **Normal**: Each line is executed in its detected language independently.
  No variable sharing across languages.
- **Auto** (``/auto``): Language is auto-detected per line. Variables stay
  within their language runtime.
- **Polyglot** (``/polyglot``): Full bidirectional bridging. Variables are
  synchronized across Python, R, and Shell after each execution.

Switch modes with slash commands:

.. code-block:: text

   /auto          # enable auto-detection
   /polyglot      # enable full bridging
   /python        # force Python mode
   /r             # force R mode
   /shell         # force Shell mode

LLM Chat Integration
--------------------

Prefix any line with ``?`` to send it to Perseus:

.. code-block:: text

   ? What test should I use for comparing two proportions?
   ? Explain the output of my last command

The ``/model`` command switches the active LLM model, and ``/models``
lists all available models.

Headless Mode
-------------

For scripting and CI pipelines, the REPL runs in headless mode:

.. code-block:: bash

   echo 'print(2+2)' | moirais repl --headless
   cat analysis_script.txt | moirais repl --headless

Headless mode reads from stdin, executes each line, prints output to
stdout, and exits when input is exhausted. All language detection and
bridging features work identically.

Callable Helpers
----------------

The REPL injects 1296 callable helpers into the Python namespace at
startup, covering all registered ``fn/`` functions. Any function can be
called directly without import:

.. code-block:: text

   [P] result = morai(values, W)
   [P] ate_result = ate(y, treatment, covariates)
   [P] alpha = crba(item_scores)

Implementation: ``runner.py`` (CLI entrypoint), ``tui.py`` (ReplScreen),
``repl_init.py`` (helper injection).
