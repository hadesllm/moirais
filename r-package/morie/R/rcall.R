# SPDX-License-Identifier: GPL-2.0-only

#' Roll-call matrix summary (Armstrong Ch 2)
#'
#' Summarises an n by m vote matrix into total yea/nay/absent counts,
#' per-roll-call marginals, and the Poole-Rosenthal "lopsided"
#' (>=97.5%) share. Accepts Poole-Rosenthal codes (1, 2, 3 = yea;
#' 4, 5, 6 = nay; 0, 7, 8, 9 = absent) and re-maps automatically.
#'
#' @param x Numeric matrix (n by m).
#' @return Named list with `n`, `m`, `n_yea`, `n_nay`, `n_abs`,
#'   `marginal_yea`, `marginal_nay`, `pct_yea`, `lopsided_pct`,
#'   `method`.
#' @export
rcall <- function(x) {
  V <- if (is.matrix(x)) x else matrix(as.numeric(x), ncol = 1L)
  V <- matrix(as.numeric(V), nrow = nrow(V), ncol = ncol(V))
  if (any(!(is.na(V) | V == 0 | V == 1))) {
    Vp <- matrix(NA_real_, nrow = nrow(V), ncol = ncol(V))
    Vp[V %in% c(1, 2, 3)] <- 1
    Vp[V %in% c(4, 5, 6)] <- 0
    V <- Vp
  }
  n <- nrow(V); m <- ncol(V)
  n_yea <- sum(V == 1, na.rm = TRUE)
  n_nay <- sum(V == 0, na.rm = TRUE)
  n_abs <- sum(is.na(V))
  marg_yea <- colSums(V == 1, na.rm = TRUE)
  marg_nay <- colSums(V == 0, na.rm = TRUE)
  denom <- marg_yea + marg_nay
  pct_yea <- ifelse(denom > 0, marg_yea / pmax(denom, 1L), NA_real_)
  lopsided <- mean((pct_yea >= 0.975) | (pct_yea <= 0.025), na.rm = TRUE)
  list(n = n, m = m, n_yea = as.integer(n_yea),
       n_nay = as.integer(n_nay), n_abs = as.integer(n_abs),
       marginal_yea = marg_yea, marginal_nay = marg_nay,
       pct_yea = pct_yea, lopsided_pct = as.numeric(lopsided),
       method = "roll_call_analysis")
}

#' @keywords internal
#' @rdname rcall
#' @export
roll_call_analysis <- rcall
