# SPDX-License-Identifier: AGPL-3.0-or-later
#' Jonckheere-Terpstra ordered-alternatives test (Gibbons & Chakraborti Ch 10.6)
#'
#' R parity for ``morie.fn.ordlt.ordered_alternatives_test``.  The
#' Python module also exports a proportional-odds ``ordered_logit``
#' estimator (kept as a separate R callable in a future release).
#'
#' Computes the Jonckheere-Terpstra statistic J = sum over (i < j) of
#' Mann-Whitney U_ij for groups with a presumed monotonic ordering,
#' then standardises to z = (J - E_J) / sqrt(Var_J) and reports a
#' two-sided p-value via the normal approximation.
#'
#' @param groups A list of numeric vectors, one per ordered group.  The
#'   list order is the assumed direction of the alternative
#'   hypothesis.
#' @return Named list with `statistic` (z), `p_value`, `J`,
#'   `E_J`, `Var_J`, `n`, `method`.
#' @references Gibbons J. D. & Chakraborti S. (2014).  Nonparametric
#'   Statistical Inference (5th ed.), Ch. 10.6.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
ordered_alternatives_test <- function(groups) {
  if (!is.list(groups) || length(groups) < 2L) {
    stop("`groups` must be a list of at least two numeric vectors")
  }
  k <- length(groups)
  ns <- vapply(groups, length, integer(1L))
  N <- sum(ns)
  if (any(ns < 1L)) stop("each group must be non-empty")

  # J = sum_{i<j} U_{ij} where U_{ij} = sum over (a in g_i, b in g_j) of
  # [I(a < b) + 0.5 * I(a == b)]
  J <- 0
  for (i in seq_len(k - 1L)) {
    gi <- groups[[i]]
    for (j in (i + 1L):k) {
      gj <- groups[[j]]
      # outer comparison
      d <- outer(gi, gj, FUN = `<`)
      e <- outer(gi, gj, FUN = `==`)
      J <- J + sum(d) + 0.5 * sum(e)
    }
  }

  # E[J] and Var[J] under H0 of common distribution (no ties)
  EJ  <- (N^2 - sum(ns^2)) / 4
  VJ  <- (N^2 * (2 * N + 3) - sum(ns^2 * (2 * ns + 3))) / 72

  z <- if (VJ > 0) (J - EJ) / sqrt(VJ) else NA_real_
  p <- if (is.finite(z)) 2 * stats::pnorm(-abs(z)) else NA_real_

  list(
    statistic = z,
    p_value   = p,
    J         = J,
    E_J       = EJ,
    Var_J     = VJ,
    n         = N,
    method    = "Jonckheere-Terpstra ordered-alternatives (normal approx)"
  )
}
