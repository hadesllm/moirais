# SPDX-License-Identifier: GPL-2.0-only

#' Bracketing number for the indicator class
#'
#' N_[](e, {1{X<=t}}, L_2(P)) = ceil(1/e^2) (Kosorok Ex 2.5.4).
#'
#' @param x Numeric vector (used only for n).
#' @param e Bracket width in L_2(P) (default 0.1).
#' @return Named list with estimate, n, method.
#' @references Kosorok (2008), Ch 2.
#' @export
ksr05_kosorok_bracketing_number <- function(x, e = 0.1) {
  x <- as.numeric(x)
  list(
    estimate = as.integer(ceiling(1 / e^2)),
    n        = length(x),
    method   = "N_[](e, {1{X<=t}}, L2(P_n)) = ceil(1/e^2)"
  )
}

# CANONICAL TEST
# ksr05_kosorok_bracketing_number(1:50, 0.1)
