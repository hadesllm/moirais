# SPDX-License-Identifier: GPL-2.0-only

#' Linear-by-linear association test for ordered categories
#' (Gibbons Ch 14.6.1)
#'
#' M^2 = (n - 1) * cor(u, v)^2 ~ chi^2_1 under independence,
#' where (u, v) are row/column scores weighted by cell counts.
#'
#' @param x r x c contingency table.
#' @param row_scores Length-r row scores; default 1..r.
#' @param col_scores Length-c col scores; default 1..c.
#' @return Named list: statistic (M^2), p_value, df, n, correlation.
#' @importFrom stats cor pchisq
#' @export
ordered_categories <- function(x, row_scores = NULL, col_scores = NULL) {
  X <- as.matrix(x); storage.mode(X) <- "numeric"
  r <- nrow(X); c <- ncol(X); n_total <- sum(X)
  if (r < 2 || c < 2 || n_total < 2) {
    return(list(statistic = NA_real_, p_value = NA_real_, df = 1L,
                n = as.integer(n_total), correlation = NA_real_,
                method = "Linear-by-linear association"))
  }
  if (is.null(row_scores)) row_scores <- seq_len(r)
  if (is.null(col_scores)) col_scores <- seq_len(c)
  # Reconstruct (u, v) pairs weighted by counts
  U <- rep(rep(row_scores, each = c), times = as.integer(t(X)))
  V <- rep(rep(col_scores, times = r), times = as.integer(t(X)))
  if (length(U) < 2 || stats::sd(U) == 0 || stats::sd(V) == 0) {
    return(list(statistic = 0, p_value = 1, df = 1L,
                n = as.integer(n_total), correlation = 0,
                method = "Linear-by-linear association"))
  }
  rho <- stats::cor(U, V)
  M2 <- (n_total - 1) * rho^2
  p <- 1 - stats::pchisq(M2, 1)
  list(
    statistic = M2,
    p_value = p,
    df = 1L,
    n = as.integer(n_total),
    correlation = rho,
    method = "Linear-by-linear (Mantel-Haenszel) trend test"
  )
}
