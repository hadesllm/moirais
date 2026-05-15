# SPDX-License-Identifier: AGPL-3.0-or-later

#' Neutral-to-the-right posterior survival (Doksum 1974).
#'
#' @param time Numeric vector of observed times.
#' @param event Optional integer/logical event indicator (1 = event, 0 = censored).
#' @param c Numeric prior concentration (default 1).
#' @param lam0 Optional baseline hazard rate.
#' @return Named list with estimate, times, S_post, H_post, c, lam0, n, method.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
ghosal_neutral_right <- function(time, event = NULL, c = 1.0, lam0 = NULL) {
  s <- .gh_surv_post(time, event, c, lam0)
  if (is.null(s)) return(list(estimate = NA_real_, n = 0,
                              method = "NTR process (empty)"))
  t_med <- stats::median(time)
  idx <- findInterval(t_med, s$times)
  est <- if (idx >= 1) s$S[idx] else 1
  list(estimate = est, times = s$times, S_post = s$S, H_post = s$H,
       c = c, lam0 = s$lam0, n = length(time),
       method = "Neutral-to-the-right posterior (Doksum 1974)")
}
