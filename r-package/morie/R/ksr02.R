# SPDX-License-Identifier: GPL-2.0-only

#' Donsker-class verification via bracketing integral
#'
#' Computes J_[](1, F, L_2(P)) = int_0^1 sqrt(log N_[](e, F, L_2(P))) de
#' for the indicator class F = {1{X<=t}}, with N_[](e) <= 2/e^2.
#'
#' @param x Numeric vector (unused, kept for API parity).
#' @return Named list with estimate, n, method.
#' @references Kosorok (2008), Ch 2 (Theorem 2.5.2).
#' @export
ksr02_kosorok_donsker_class <- function(x) {
  x <- as.numeric(x)
  integrand <- function(e) sqrt(log(2) - 2 * log(e))
  j <- stats::integrate(integrand, lower = 1e-8, upper = 1.0,
                        subdivisions = 200L)$value
  list(
    estimate = j,
    n        = length(x),
    method   = "Bracketing-integral Donsker verification (indicator class)"
  )
}

# CANONICAL TEST
# ksr02_kosorok_donsker_class(1:10)
