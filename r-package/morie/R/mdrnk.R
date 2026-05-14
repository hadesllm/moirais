# SPDX-License-Identifier: MIT OR Apache-2.0

#' Midrank vector with tie summary (Gibbons Ch 5.6.2)
#'
#' Identical to `rank(x, ties.method = "average")` plus a tie-
#' correction term `sum t_j^3 - t_j` over tied groups.
#'
#' @param x Numeric vector.
#' @return Named list: midranks, n, ties, tie_correction.
#' @export
midranks <- function(x) {
  x <- as.numeric(x)
  n <- length(x)
  if (n < 1) {
    return(list(midranks = numeric(0), n = 0L, ties = list(),
                tie_correction = 0,
                method = "Midranks"))
  }
  mr <- rank(x, ties.method = "average")
  tab <- table(x)
  ties <- Filter(function(z) z[[2]] > 1, Map(function(v, c) list(as.numeric(v), as.integer(c)),
                                              names(tab), tab))
  tie_correction <- sum(sapply(tab[tab > 1], function(c) c^3 - c))
  if (length(tab[tab > 1]) == 0) tie_correction <- 0
  list(
    midranks = mr,
    n = n,
    ties = ties,
    tie_correction = as.numeric(tie_correction),
    method = "Midranks (Gibbons 5.6.2)"
  )
}
