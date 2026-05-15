# SPDX-License-Identifier: AGPL-3.0-or-later

#' One-step efficient location estimator
#'
#' theta_tilde = theta_init + (1/n) * sum IF(X_i; theta_init).
#'
#' @param x Numeric vector.
#' @param y Ignored (API parity).
#' @return Named list with estimate, se, n, method.
#' @references Kosorok (2008), Ch 7.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
ksr15_kosorok_one_step_estimator <- function(x, y = NULL) {
  x <- as.numeric(x)
  n <- length(x)
  theta_init <- stats::median(x)
  IF <- x - theta_init
  theta_tilde <- theta_init + mean(IF)
  se <- stats::sd(x) / sqrt(n)
  list(
    estimate = theta_tilde, se = se, n = n,
    method   = "One-step from median: theta + mean(x-theta)"
  )
}

# CANONICAL TEST
# set.seed(0); ksr15_kosorok_one_step_estimator(rnorm(200))

#' @rdname ksr15_kosorok_one_step_estimator
#' @keywords internal
#' @export
kosorok_one_step_estimator <- ksr15_kosorok_one_step_estimator
