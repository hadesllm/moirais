# SPDX-License-Identifier: GPL-2.0-only

#' Donsker-class verification via bracketing integral
#'
#' Computes J_[](1, F, L_2(P)) = int_0^1 sqrt(log N_brackets(e, F, L_2(P))) de
#' for the indicator class F of one-sided thresholds on X (Kosorok Ex 2.5.4), with bracketing number bounded by 2 over epsilon squared.
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

#' @rdname ksr02_kosorok_donsker_class
#' @keywords internal
#' @export
kosorok_donsker_class <- ksr02_kosorok_donsker_class
