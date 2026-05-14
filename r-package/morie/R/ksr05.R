# SPDX-License-Identifier: MIT OR Apache-2.0

#' Bracketing number for the indicator class
#'
#' "bracketing number for one-sided threshold class"
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
    method   = "bracketing number for one-sided threshold class (Kosorok 2.5.4)"
  )
}

# CANONICAL TEST
# ksr05_kosorok_bracketing_number(1:50, 0.1)

#' @rdname ksr05_kosorok_bracketing_number
#' @keywords internal
#' @export
kosorok_bracketing_number <- ksr05_kosorok_bracketing_number
