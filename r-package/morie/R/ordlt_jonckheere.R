# SPDX-License-Identifier: GPL-2.0-only

#' Jonckheere-Terpstra ordered-alternatives test (Gibbons Ch 10.6)
#'
#' Tests H0: F_1 = ... = F_k against the ordered alternative
#' H1: F_1 <= F_2 <= ... <= F_k.  J = sum over i<j of U_ij (Mann-Whitney
#' counts with 1/2 weight for ties).
#'
#' Normal approximation:
#'   E_J = (N^2 - sum n_i^2) / 4
#'   Var_J = (N^2 (2N + 3) - sum n_i^2 (2 n_i + 3)) / 72
#'
#' @param groups List of numeric vectors in monotone hypothesised order.
#' @return Named list: statistic, p_value, z, E_J, Var_J, n, k, method.
#' @importFrom stats pnorm
#' @export
ordered_alternatives_test <- function(groups) {
  if (!is.list(groups) || length(groups) < 2)
    return(list(statistic = NA_real_, p_value = NA_real_, z = NA_real_,
                n = 0L, k = length(groups),
                method = "Jonckheere-Terpstra ordered-alternatives test"))
  arrs <- lapply(groups, as.numeric)
  k <- length(arrs)
  J <- 0
  for (i in 1:(k - 1)) {
    for (j in (i + 1):k) {
      ai <- arrs[[i]]; aj <- arrs[[j]]
      # Vectorise: for each ai count #(aj > ai) + 0.5 * #(aj == ai)
      lt <- sum(outer(ai, aj, "<"))
      eq <- sum(outer(ai, aj, "=="))
      J <- J + lt + 0.5 * eq
    }
  }
  ns <- sapply(arrs, length)
  N <- sum(ns)
  E_J <- (N^2 - sum(ns^2)) / 4
  Var_J <- (N^2 * (2 * N + 3) - sum(ns^2 * (2 * ns + 3))) / 72
  z <- (J - E_J) / sqrt(Var_J)
  p <- 2 * (1 - stats::pnorm(abs(z)))
  list(
    statistic = J,
    p_value = p,
    z = z,
    E_J = E_J,
    Var_J = Var_J,
    n = N,
    k = k,
    method = "Jonckheere-Terpstra ordered-alternatives test"
  )
}
