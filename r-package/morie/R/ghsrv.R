# SPDX-License-Identifier: GPL-2.0-only

#' Beta-process posterior survival (Hjort 1990).
#'
#' @param time Numeric vector of observed times.
#' @param event Optional integer/logical event indicator (1 = event, 0 = censored).
#' @param c Numeric prior concentration (default 1).
#' @param lam0 Optional baseline hazard rate.
#' @return Named list with estimate, times, S_post, H_post, c, lam0, n, method.
#' @export
ghosal_survival_beta_process <- function(time, event = NULL, c = 1.0,
                                          lam0 = NULL) {
  s <- .gh_surv_post(time, event, c, lam0)
  if (is.null(s)) return(list(estimate = NA_real_, n = 0,
                              method = "Beta-process survival (empty)"))
  t_med <- stats::median(time)
  idx <- findInterval(t_med, s$times)
  est <- if (idx >= 1) s$S[idx] else 1
  list(estimate = est, times = s$times, S_post = s$S, H_post = s$H,
       c = c, lam0 = s$lam0, n = length(time),
       method = "Beta-process posterior survival (Hjort 1990)")
}
