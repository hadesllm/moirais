# SPDX-License-Identifier: MIT OR Apache-2.0

#' Adaptive contraction rates over a smoothness grid.
#'
#' @param x Numeric data vector (used only for sample-size n).
#' @param betas Numeric vector of smoothness exponents (default seq(0.5, 3, length.out = 11)).
#' @param d Integer dimension (default 1).
#' @return Named list with estimate, betas, rates, best_beta, n, d, method.
#' @export
ghosal_adaptation <- function(x, betas = NULL, d = 1) {
  n <- length(x)
  if (is.null(betas)) betas <- seq(0.5, 3.0, length.out = 11)
  rates <- n^(-betas / (2*betas + d))
  best <- which.min(rates)
  list(estimate = rates[best], betas = betas, rates = rates,
       best_beta = betas[best], n = n, d = d,
       method = "Adaptive posterior contraction over Holder grid")
}
