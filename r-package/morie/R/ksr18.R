# SPDX-License-Identifier: MIT OR Apache-2.0

#' Nelson-Aalen cumulative hazard at the largest event time
#'
#' Lambda_hat(t) = sum over t_i <= t of d_i / Y_i.  Variance: sum d_i/Y_i^2.
#'
#' @param t Numeric vector of observed times.
#' @param event Integer/logical vector (1 = event, 0 = censored).
#' @return Named list with estimate, se, n, method.
#' @references Kosorok (2008), Ch 8.
#' @export
ksr18_kosorok_nelson_aalen <- function(t, event) {
  t <- as.numeric(t); event <- as.integer(event)
  n <- length(t)
  ord <- order(t)
  t <- t[ord]; event <- event[ord]
  cum_h <- 0; cum_v <- 0
  i <- 1L
  while (i <= n) {
    j <- i
    while (j <= n && t[j] == t[i]) j <- j + 1L
    d <- sum(event[i:(j - 1L)])
    Y <- n - i + 1L
    if (d > 0) {
      cum_h <- cum_h + d / Y
      cum_v <- cum_v + d / (Y^2)
    }
    i <- j
  }
  list(
    estimate = cum_h,
    se       = sqrt(cum_v),
    n        = n,
    method   = "Nelson-Aalen cumulative hazard at t_max"
  )
}

# CANONICAL TEST
# ksr18_kosorok_nelson_aalen(1:10, c(1,1,0,1,1,0,1,1,1,0))

#' @rdname ksr18_kosorok_nelson_aalen
#' @keywords internal
#' @export
kosorok_nelson_aalen <- ksr18_kosorok_nelson_aalen
