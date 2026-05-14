# SPDX-License-Identifier: MIT OR Apache-2.0

#' Glivenko-Cantelli theorem verification (one-sample KS)
#'
#' sup_t |F_n(t) - F(t)| under a hypothesised CDF F.  By
#' Glivenko-Cantelli the statistic -> 0 a.s. when F is correct.
#'
#' @param x Numeric vector.
#' @param cdf Name of CDF (default \"pnorm\").
#' @return Named list with statistic, p_value, n, method.
#' @references Kosorok (2008), Ch 2.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
ksr03_kosorok_glivenko_cantelli <- function(x, cdf = "pnorm") {
  x <- as.numeric(x)
  n <- length(x)
  res <- suppressWarnings(stats::ks.test(x, cdf))
  list(
    statistic = unname(res$statistic),
    p_value   = res$p.value,
    n         = n,
    method    = "Glivenko-Cantelli / KS sup|F_n - F|"
  )
}

# CANONICAL TEST
# set.seed(0); ksr03_kosorok_glivenko_cantelli(rnorm(200))

#' @rdname ksr03_kosorok_glivenko_cantelli
#' @keywords internal
#' @export
kosorok_glivenko_cantelli <- ksr03_kosorok_glivenko_cantelli
