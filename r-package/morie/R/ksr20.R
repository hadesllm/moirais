# SPDX-License-Identifier: AGPL-3.0-or-later

#' Kaplan-Meier estimator of the censoring distribution
#'
#' S_C(t) by KM on (t_i, 1 - delta_i).  Greenwood SE.
#'
#' @param t Numeric vector of observed times.
#' @param event Integer/logical (1 = survival event, 0 = censored).
#' @return Named list with estimate, se, n, method.
#' @references Kosorok (2008), Ch 8.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
ksr20_kosorok_censoring_survival <- function(t, event) {
  t <- as.numeric(t); event <- as.integer(event)
  n <- length(t)
  c_indic <- 1L - event
  ord <- order(t)
  t <- t[ord]; c_indic <- c_indic[ord]
  S <- 1; var_factor <- 0
  i <- 1L
  while (i <= n) {
    j <- i
    while (j <= n && t[j] == t[i]) j <- j + 1L
    d <- sum(c_indic[i:(j - 1L)])
    Y <- n - i + 1L
    if (d > 0) {
      S <- S * (1 - d / Y)
      if (Y > d) var_factor <- var_factor + d / (Y * (Y - d))
    }
    i <- j
  }
  list(
    estimate = S,
    se       = S * sqrt(var_factor),
    n        = n,
    method   = "Kaplan-Meier of censoring (Greenwood SE) at t_max"
  )
}

# CANONICAL TEST
# ksr20_kosorok_censoring_survival(1:10, c(1,1,0,1,1,0,1,1,1,0))

#' @rdname ksr20_kosorok_censoring_survival
#' @keywords internal
#' @export
kosorok_censoring_survival <- ksr20_kosorok_censoring_survival
