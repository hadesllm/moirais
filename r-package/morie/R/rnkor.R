# SPDX-License-Identifier: MIT OR Apache-2.0

#' Signed ranks of paired differences (Gibbons Ch 5.5)
#'
#' Signed ranks R_i^+ = sign(D_i) * rank(|D_i|) used by Wilcoxon
#' signed-rank.
#'
#' @param x Numeric vector of differences (or values; mu0 is subtracted).
#' @param mu0 Hypothesised median (default 0).
#' @return Named list: signed_ranks, abs_ranks, W_plus, W_minus, n_nonzero, n.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
rank_order_statistics <- function(x, mu0 = 0) {
  x <- as.numeric(x)
  n <- length(x)
  if (n < 2) {
    return(list(signed_ranks = numeric(0), abs_ranks = numeric(0),
                W_plus = NA_real_, W_minus = NA_real_,
                n_nonzero = 0L, n = n,
                method = "Rank-order signed ranks"))
  }
  d <- x - mu0
  nz <- d != 0
  signed_ranks <- numeric(n)
  abs_ranks <- numeric(n)
  if (any(nz)) {
    ar <- rank(abs(d[nz]))
    signed_ranks[nz] <- sign(d[nz]) * ar
    abs_ranks[nz] <- ar
  }
  W_plus <- sum(signed_ranks[signed_ranks > 0])
  W_minus <- -sum(signed_ranks[signed_ranks < 0])
  list(
    signed_ranks = signed_ranks,
    abs_ranks = abs_ranks,
    W_plus = W_plus,
    W_minus = W_minus,
    n_nonzero = sum(nz),
    n = n,
    method = "Rank-order signed ranks"
  )
}
