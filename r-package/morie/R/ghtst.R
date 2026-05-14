# SPDX-License-Identifier: MIT OR Apache-2.0

#' Polya-tree Bayes factor for H0: F = N(loc, scale^2).
#'
#' @param x Numeric data vector.
#' @param ref_loc Numeric reference location (default 0).
#' @param ref_scale Numeric reference scale (default 1).
#' @param depth Integer Polya-tree depth (default 6).
#' @param c Numeric Polya-tree concentration (default 1).
#' @return Named list with statistic (log BF), p_value, BF10, log_BF10, n, depth, method.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
ghosal_np_testing <- function(x, ref_loc = 0, ref_scale = 1, depth = 6,
                                c = 1.0) {
  x <- as.numeric(x); n <- length(x)
  if (n < 2) return(list(statistic = NA_real_, p_value = NA_real_, n = n,
                          method = "Polya-tree BF (n<2)"))
  u <- stats::pnorm(x, mean = ref_loc, sd = ref_scale)
  log_bf <- 0
  for (m in seq_len(depth)) {
    nbins <- 2^m
    edges <- seq(0, 1, length.out = nbins + 1)
    bin <- findInterval(u, edges, rightmost.closed = TRUE)
    bin <- pmin(pmax(bin, 1L), nbins)
    counts <- tabulate(bin, nbins = nbins)
    alpha <- c * m * m
    n0 <- counts[seq(1, nbins, by = 2)]; n1 <- counts[seq(2, nbins, by = 2)]
    log_bf <- log_bf + sum(lbeta(alpha + n0, alpha + n1) - lbeta(alpha, alpha))
  }
  BF10 <- exp(log_bf)
  p_value <- if (BF10 > 1) 1 / (1 + BF10) else 0.5
  list(statistic = log_bf, p_value = p_value, BF10 = BF10,
       log_BF10 = log_bf, n = n, depth = depth,
       method = "Polya-tree Bayes-factor test (Berger-Guglielmi)")
}
