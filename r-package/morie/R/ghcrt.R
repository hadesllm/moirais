# SPDX-License-Identifier: MIT OR Apache-2.0

#' Minimax posterior-contraction rate
#'
#' Returns eps_n = n raised to the power of -beta/(2*beta+d).
#'
#' @param x Numeric data vector (used only for sample-size n).
#' @param beta Numeric smoothness exponent (default 1.0).
#' @param d Integer dimension (default 1).
#' @return Named list with estimate, log_rate_correction, parametric_rate, n, beta, d, method.
#' @export
ghosal_contraction_rate <- function(x, beta = 1.0, d = 1) {
  n <- length(x)
  if (n <= 1) return(list(estimate = NA_real_, n = n,
                          method = "Contraction rate (n too small)"))
  eps_n <- n^(-beta / (2 * beta + d))
  list(estimate = eps_n,
       log_rate_correction = (log(n))^(beta/(2*beta+d)) * eps_n,
       parametric_rate = n^(-0.5),
       n = n, beta = beta, d = d,
       method = "Minimax contraction rate n^{-beta/(2beta+d)}")
}
